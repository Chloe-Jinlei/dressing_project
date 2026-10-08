"""Genesis simulation worker for bunnylab's canonical Route A policy/dashboard.

The JSON bridge exports only RGB, camera calibration, and robot poses. Meshes,
material settings, particle state, and contact details stay in private outputs.
"""
import contextlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def array(value):
    return value.detach().cpu().numpy().reshape(-1)


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()


class GenesisController:
    """Robot-only IK plus contact-based, local no-slip grasp approximation."""
    tcp_offset = np.array([0., 0., .1034])

    def __init__(self, scene, robots, objects, cameras, config, report, run):
        self.scene=scene;self.robots=robots;self.objects=objects;self.cameras=cameras
        self.config=config;self.report=report;self.run=run
        self.dt=report["genesis"]["step_dt_s"];self.steps=0
        self.hands={side:robot.get_link("hand") for side,robot in robots.items()}
        self.fingers={side:[robot.get_link(name) for name in ("left_finger","right_finger")]
                      for side,robot in robots.items()}
        self.controls=np.concatenate([array(robots[side].get_qpos()) for side in ("left","right")])
        self.attached={};self.used=set();self.closing={side:False for side in robots}
        self.contact_log=(run/"grasp_events.jsonl").open("w")
        with np.load(run/"assets"/"physics_meshes.npz",allow_pickle=False) as physics:
            from dressing_scene import tetra_boundary
            self.surface_indices={"sloth":tetra_boundary(physics["sloth_points"],physics["sloth_tetra"])[0],
                                  "garment":np.unique(physics["garment_faces"])}
        for robot in robots.values():
            if robot.n_qs!=9 or robot.n_dofs!=9:
                raise RuntimeError("Genesis Route A requires the supplied fixed-base, nine-DOF Panda asset")

    @property
    def time(self):return self.steps*self.dt

    def observe(self):
        robot_state={}
        for side,robot in self.robots.items():
            hand=self.hands[side];quat=array(hand.get_quat(relative=False))
            position=array(hand.get_pos(relative=False))+rotation(quat)@self.tcp_offset
            qpos=array(robot.get_qpos())
            if not np.isfinite(qpos).all():raise RuntimeError("Robot state is non-finite")
            robot_state[side+"_position_m"]=position.tolist()
            robot_state[side+"_quaternion_wxyz"]=quat.tolist()
            robot_state[side+"_gripper"]=float(np.clip(qpos[7:].sum()/.08,0,1))
        calibration={}
        self.cameras["wrist"].move_to_attach()
        for name in ("overhead","wrist","table_side"):
            camera=self.cameras[name]
            transform=camera.get_transform().detach().cpu().numpy().reshape(4,4)
            calibration[name]={"position_world_m":transform[:3,3].tolist(),
                "rotation_camera_to_world":transform[:3,:3].tolist(),
                "vertical_fov_deg":float(camera.fov),
                "image_width":camera.res[0],"image_height":camera.res[1],
                "axes":"camera +X right, +Y up, viewing direction -Z"}
        return {"robot":robot_state,"camera_calibration":calibration}

    def result(self,**payload):
        return {"time":self.time,"observation":self.observe(),**payload}

    def capture(self):
        for name in ("overhead","wrist","table_side"):
            if name=="wrist":self.cameras[name].move_to_attach()
            rgb,*_=self.cameras[name].render()
            Image.fromarray(np.asarray(rgb)).save(self.run/(name+".png"))
        return self.result()

    def plan(self,action):
        control=self.config["control"]
        expected={side+suffix for side in ("left","right")
                  for suffix in ("_delta_m","_rotation_rad","_gripper")}
        if set(action)!=expected:raise ValueError("Action fields do not match the Route A schema")
        current=self.observe()["robot"];values=self.controls.copy();targets={}
        for side,start in (("left",0),("right",9)):
            delta=np.asarray(action[side+"_delta_m"],dtype=float)
            rot=np.asarray(action[side+"_rotation_rad"],dtype=float)
            opening=float(action[side+"_gripper"])
            if delta.shape!=(3,) or rot.shape!=(3,) or not np.isfinite(np.r_[delta,rot,opening]).all():
                raise ValueError("Action must contain finite three-dimensional vectors")
            if np.linalg.norm(delta)>control["max_translation_m"]+1e-12:
                raise ValueError("Translation exceeds allowed norm")
            if np.linalg.norm(rot)>control["max_rotation_rad"]+1e-12 or not 0<=opening<=1:
                raise ValueError("Rotation or gripper opening exceeds limits")
            target=np.asarray(current[side+"_position_m"])+delta
            if np.any(target<control["workspace_min"]) or np.any(target>control["workspace_max"]):
                raise ValueError("Target outside declared robot workspace")
            robot=self.robots[side];original=array(robot.get_qpos())
            if np.linalg.norm(delta)+np.linalg.norm(rot)>1e-12:
                old_quat=np.asarray(current[side+"_quaternion_wxyz"])
                quaternion=(Rotation.from_rotvec(rot)*Rotation.from_quat(old_quat[[1,2,3,0]])).as_quat()[[3,0,1,2]]
                qpos,error=robot.inverse_kinematics(
                    link=self.hands[side],pos=target,quat=quaternion,local_point=self.tcp_offset,
                    init_qpos=robot.get_qpos(),dofs_idx_local=list(range(7)),
                    max_samples=1,max_solver_iters=100,pos_tol=.0005,rot_tol=.005,return_error=True)
                error=array(error);qpos=array(qpos)
                if not np.isfinite(qpos).all() or np.linalg.norm(error[:3])>.001 or np.linalg.norm(error[3:])>.01:
                    raise ValueError(f"{side} IK did not converge; action rejected")
                if np.max(np.abs(qpos[:7]-original[:7]))>control["max_joint_delta_rad"]:
                    raise ValueError("IK joint displacement exceeds control limit")
                values[start:start+7]=qpos[:7]
            # Gripper aperture is a control target, never a position reset.
            values[start+7:start+9]=opening*.04
            targets[side]=target.tolist()
            if not np.array_equal(original,array(robot.get_qpos())):
                raise RuntimeError("IK unexpectedly changed the simulated robot state")
        return {"controls":values.tolist(),"targets":targets}

    def log_grasp(self,event,side,**extra):
        self.contact_log.write(json.dumps({"event":event,"side":side,"time":self.time,**extra})+"\n")
        self.contact_log.flush()

    def release(self,side):
        binding=self.attached.pop(side,None)
        if binding is None:return
        name=binding["object"]
        for indices in binding["indices"]:
            self.objects[name].release_particle(particles_idx_local=indices)
            self.used.difference_update((name,int(idx)) for idx in indices)
        self.log_grasp("released",side,object=name)

    def set_control(self,values):
        values=np.asarray(values,dtype=float)
        if values.shape!=(18,) or not np.isfinite(values).all():raise ValueError("Invalid Genesis joint controls")
        self.controls=values.copy()
        for side,start in (("left",0),("right",9)):
            command=values[start:start+9].copy()
            binding=self.attached.get(side)
            if binding and command[7:].max()>max(binding["finger_positions"])+.002:
                self.release(side);binding=None
            self.closing[side]=command[7:].max()<.015
            if binding:command[7:]=binding["finger_positions"]
            self.robots[side].control_dofs_position(command)

    def grasp_contacts(self):
        """Only actual closed-gripper pad contacts activate a local attachment."""
        for side in self.robots:
            if not self.closing[side] or side in self.attached:continue
            qpos=array(self.robots[side].get_qpos())
            if qpos[7:].max()>.03:continue
            for name,entity in self.objects.items():
                points=entity.get_particles_pos().detach().cpu().numpy()
                pads=[]
                for finger in self.fingers[side]:
                    local=(points-array(finger.get_pos(relative=False)))@rotation(array(finger.get_quat(relative=False)))
                    d=np.abs(local-[0,.0055,.0445])-[.0085,.004,.0085]
                    distance=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(axis=1),0)
                    candidates=self.surface_indices[name]
                    candidates=candidates[distance[candidates]<=.004]
                    candidates=np.array([idx for idx in candidates if (name,int(idx)) not in self.used],dtype=int)
                    pads.append(candidates[np.argsort(distance[candidates])[:4]])
                if not all(len(indices) for indices in pads):continue
                # Do not attach a node to two fingers or two arms.
                pads[1]=np.setdiff1d(pads[1],pads[0])
                if not len(pads[1]):continue
                for finger,indices in zip(self.fingers[side],pads):
                    entity.fix_particles_to_link(finger.idx,particles_idx_local=indices)
                    self.used.update((name,int(idx)) for idx in indices)
                self.attached[side]={"object":name,"indices":pads,"finger_positions":qpos[7:].tolist()}
                self.log_grasp("attached",side,object=name,indices=[indices.tolist() for indices in pads])
                self.set_control(self.controls)
                break

    def step(self,count):
        if type(count) is not int or not 1<=count<=1000:raise ValueError("Invalid simulation step count")
        for _ in range(count):
            self.scene.step();self.steps+=1
            if self.steps%4==0:self.grasp_contacts()
        for entity in self.objects.values():
            if not np.isfinite(entity.get_particles_pos().detach().cpu().numpy()).all():
                raise RuntimeError("Genesis deformable state is non-finite")
        return self.result()

    def hold(self):
        self.set_control(np.concatenate([array(self.robots[side].get_qpos()) for side in ("left","right")]))
        return self.result(controls=self.controls.tolist())


def serve_scene(args):
    from dressing_scene import HERE, prepare_assets, build_scene, positions_stats
    protocol=sys.stdout
    def reply(serial,ok,result=None,error=None):
        payload={"id":serial,"ok":ok,"result":result} if ok else {
            "id":serial,"ok":False,"kind":type(error).__name__,"error":str(error)}
        protocol.write("DRESSING_RPC "+json.dumps(payload,allow_nan=False)+"\n");protocol.flush()
    run=Path(args.output).resolve();run.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault("GS_CACHE_FILE_PATH",str(HERE/".cache"/"genesis"))
    os.environ.setdefault("XDG_CACHE_HOME",str(HERE/".cache"))
    os.environ.setdefault("MPLCONFIGDIR",str(HERE/".cache"/"matplotlib"))
    controller=None;report={"status":"starting"}
    with contextlib.redirect_stdout(sys.stderr):
        try:
            config,paths,prepared,report=prepare_assets(args.source_project.resolve(),run)
            report["limitations"][0]="Route A scene: bounded visual policy actions; dressing success is judged by the user."
            report["limitations"].append("Closed-gripper pad contact activates local no-slip PBD attachments; reciprocal robot load and frictional grasp force are not validated.")
            scene,robots,objects,cameras=build_scene(config,paths,prepared,report,args)
            controller=GenesisController(scene,robots,objects,cameras,config,report,run)
            # Standard setup settling; no scripted object manipulation or policy.
            report["settling_history"]=[{"simulation_seconds":controller.time,
                "objects":positions_stats(objects,config["scene"]["table_height"])}]
            for i in range(args.steps):
                scene.step();controller.steps+=1
                if (i+1)%50==0:
                    report["settling_history"].append({"simulation_seconds":controller.time,
                        "objects":positions_stats(objects,config["scene"]["table_height"])})
                if (i+1)%100==0:print(f"Genesis settling {i+1}/{args.steps}",flush=True)
            initial=positions_stats(objects,config["scene"]["table_height"])
            if any(state["table_penetration_m"]>.0001 for state in initial.values()):
                raise RuntimeError("Genesis settling failed: particles penetrated the tabletop")
            report.update(status="route_a_ready",initial_state=initial)
            (run/"report.json").write_text(json.dumps(report,indent=2))
            reply(0,True,controller.result(timestep=controller.dt,controls=controller.controls.tolist(),limitations=report["limitations"]))
            for line in sys.stdin:
                request=json.loads(line);serial=request["id"]
                try:
                    op=request["op"]
                    if op=="capture":result=controller.capture()
                    elif op=="plan":result=controller.plan(request["action"])
                    elif op=="control":controller.set_control(request["values"]);result={}
                    elif op=="step":result=controller.step(request["count"])
                    elif op=="hold":result=controller.hold()
                    elif op=="close":reply(serial,True,{});break
                    else:raise ValueError("Unknown Genesis operation")
                    reply(serial,True,result)
                except ValueError as exc:reply(serial,False,error=exc)
                except Exception as exc:
                    traceback.print_exc();reply(serial,False,error=exc);break
            report.update(status="route_a_closed",simulation_seconds=controller.time,
                          final_state=positions_stats(objects,config["scene"]["table_height"]))
        except Exception as exc:
            traceback.print_exc();report.update(status="failed",error=str(exc))
            if controller is None:reply(0,False,error=exc)
            raise
        finally:
            (run/"report.json").write_text(json.dumps(report,indent=2))
            if controller:controller.contact_log.close()
            if "genesis" in sys.modules:sys.modules["genesis"].destroy()
