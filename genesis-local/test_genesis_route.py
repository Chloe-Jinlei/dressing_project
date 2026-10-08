"""Control checks and opt-in CPU contact integration; no network/model calls."""
import io
import json
import os
from pathlib import Path
import tomllib
import unittest
from unittest.mock import Mock

import numpy as np
from genesis_route import GenesisController


class Tensor:
    def __init__(self,value):self.value=np.asarray(value,dtype=float)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value.copy()


class Finger:
    def __init__(self,idx,pos,quat):self.idx=idx;self.pos=pos;self.quat=quat
    def get_pos(self,**kwargs):return Tensor(self.pos)
    def get_quat(self,**kwargs):return Tensor(self.quat)


def controller_fixture():
    controller=GenesisController.__new__(GenesisController)
    controller.steps=0;controller.dt=.005;controller.attached={};controller.used=set()
    controller.contact_log=io.StringIO()
    controller.controls=np.r_[np.zeros(7),[0.,0.],np.zeros(7),[.04,.04]]
    controller.closing={"left":True,"right":False}
    controller.robots={side:Mock() for side in ("left","right")}
    for robot in controller.robots.values():
        robot.get_qpos.return_value=Tensor(np.r_[np.zeros(7),[.01,.01]])
    controller.fingers={
        "left":[Finger(10,[0,-.018,0],[1,0,0,0]),Finger(11,[0,.018,0],[0,0,0,1])],
        "right":[Finger(20,[1,-.018,0],[1,0,0,0]),Finger(21,[1,.018,0],[0,0,0,1])],
    }
    x=np.array([-.004,.004,0.,.006])
    points=np.concatenate([np.column_stack([x,np.full(4,-.0085),np.full(4,.0445)]),
                           np.column_stack([x,np.full(4,.0085),np.full(4,.0445)])])
    entity=Mock();entity.get_particles_pos.return_value=Tensor(points)
    controller.objects={"garment":entity};controller.surface_indices={"garment":np.arange(8)}
    return controller,entity,points


class ContactGateTests(unittest.TestCase):
    def test_remote_or_single_pad_contact_does_not_attach(self):
        controller,entity,points=controller_fixture()
        entity.get_particles_pos.return_value=Tensor(points+[0,.3,0])
        controller.grasp_contacts();entity.fix_particles_to_link.assert_not_called()
        points[4:]+=np.array([0,.3,0])
        entity.get_particles_pos.return_value=Tensor(points)
        controller.grasp_contacts();entity.fix_particles_to_link.assert_not_called()

    def test_open_command_does_not_attach_even_with_pad_contact(self):
        controller,entity,_=controller_fixture()
        values=controller.controls.copy();values[7:9]=.04
        controller.set_control(values);controller.grasp_contacts()
        entity.fix_particles_to_link.assert_not_called()

    def test_both_pad_contacts_attach_disjoint_local_nodes_and_open_releases(self):
        controller,entity,_=controller_fixture()
        controller.grasp_contacts()
        calls=entity.fix_particles_to_link.call_args_list
        self.assertEqual(len(calls),2)
        groups=[set(call.kwargs["particles_idx_local"]) for call in calls]
        self.assertTrue(all(0<len(group)<=4 for group in groups))
        self.assertFalse(groups[0]&groups[1])
        self.assertEqual(len(controller.used),8)
        # The contact aperture is held until an opening command is received.
        np.testing.assert_allclose(controller.robots["left"].control_dofs_position.call_args.args[0][7:],[.01,.01])
        values=controller.controls.copy();values[7:9]=.04
        controller.set_control(values)
        self.assertFalse(controller.attached);self.assertFalse(controller.used)
        self.assertEqual(entity.release_particle.call_count,2)
        self.assertIn('"event": "released"',controller.contact_log.getvalue())

    def test_ik_branch_jump_is_rejected_without_sending_controls(self):
        controller,entity,_=controller_fixture()
        controller.config=tomllib.loads((Path(__file__).resolve().parent.parent/"bunnylab"/"config.toml").read_text())
        controller.hands={side:object() for side in controller.robots}
        controller.observe=lambda:{"robot":{
            "left_position_m":[0,0,1],"right_position_m":[.1,0,1],
            "left_quaternion_wxyz":[1,0,0,0],"right_quaternion_wxyz":[1,0,0,0]}}
        q=np.r_[np.zeros(7),[.01,.01]];q[0]=.21
        controller.robots["left"].inverse_kinematics.return_value=(Tensor(q),Tensor(np.zeros(6)))
        action={"left_delta_m":[.01,0,0],"right_delta_m":[0,0,0],
            "left_rotation_rad":[0,0,0],"right_rotation_rad":[0,0,0],"left_gripper":1,"right_gripper":1}
        with self.assertRaisesRegex(ValueError,"joint displacement"):
            controller.plan(action)
        controller.robots["left"].control_dofs_position.assert_not_called()


@unittest.skipUnless(os.environ.get("GENESIS_CONTACT_TEST_RUN_DIR"), "opt-in actual CPU contact integration")
class RigidPBDFrictionTests(unittest.TestCase):
    def test_supported_particles_stop_slide_and_remain_free_in_flight(self):
        import genesis as gs
        import trimesh
        from genesis_mesh_adapter import existing_pbd_mesh, install_rigid_pbd_friction

        run = Path(os.environ["GENESIS_CONTACT_TEST_RUN_DIR"])
        run.mkdir(parents=True, exist_ok=True)
        gs.init(backend=gs.cpu, seed=0, precision="32", logging_level="warning")
        try:
            scene = gs.Scene(show_viewer=False,
                sim_options=gs.options.SimOptions(dt=.001, substeps=1),
                pbd_options=gs.options.PBDOptions(particle_size=.004,
                    lower_bound=(-1, -1, -1), upper_bound=(1, 1, 1)))
            contact = install_rigid_pbd_friction(scene)
            scene.add_entity(morph=gs.morphs.Plane(),
                material=gs.materials.Rigid(needs_coup=True, coup_friction=.7))
            cases = {"frictionless": (0., .1, .002), "static": (.6, .001, .002),
                     "sliding": (.6, .4, .002), "airborne": (.6, .1, .2)}
            faces = np.array([[a, b, c] for row in range(3) for col in range(3)
                for a, b, c in ((row * 4 + col, row * 4 + col + 1, (row + 1) * 4 + col + 1),
                                (row * 4 + col, (row + 1) * 4 + col + 1, (row + 1) * 4 + col))], dtype=np.int32)
            objects = {}
            for index, (name, (friction, speed, height)) in enumerate(cases.items()):
                points = np.array([[col * .01, row * .01, height] for row in range(4) for col in range(4)])
                points[:, 0] += -.3 + index * .15
                mesh = run / f"{name}.obj"
                trimesh.Trimesh(points, faces, process=False).export(mesh)
                with existing_pbd_mesh("garment", points, faces, .001):
                    objects[name] = scene.add_entity(
                        morph=gs.morphs.Mesh(file=str(mesh), file_meshes_are_zup=True,
                                             decimate=False, convexify=False),
                        material=gs.materials.PBD.Cloth(static_friction=friction,
                            kinetic_friction=friction, air_resistance=0))
            scene.build()
            initial = {}
            for name, entity in objects.items():
                initial[name] = entity.get_particles_pos().cpu().numpy().copy()
                entity.set_particles_vel(np.tile([cases[name][1], 0., 0.], (entity.n_particles, 1)))
            checkpoints = {}
            for step in range(100):
                scene.step()
                if step + 1 in (1, 20, 100):
                    checkpoints[str(step + 1)] = {name: {
                        "mean_velocity": entity.get_particles_vel().cpu().numpy().mean(0).tolist(),
                        "mean_displacement": (entity.get_particles_pos().cpu().numpy() - initial[name]).mean(0).tolist(),
                    } for name, entity in objects.items()}
            (run / "report.json").write_text(json.dumps({"backend": "cpu", "real_model_call": False,
                "contact": contact, "checkpoints": checkpoints}, indent=2))
            early, late = checkpoints["20"], checkpoints["100"]
            self.assertAlmostEqual(late["frictionless"]["mean_velocity"][0], .1, delta=.002)
            self.assertLess(abs(late["static"]["mean_displacement"][0]), 2e-6)
            self.assertGreater(early["sliding"]["mean_velocity"][0], .25)
            self.assertLess(early["sliding"]["mean_velocity"][0], .33)
            self.assertLess(abs(late["sliding"]["mean_velocity"][0]), .002)
            self.assertGreater(late["sliding"]["mean_displacement"][0], .01)
            self.assertAlmostEqual(late["airborne"]["mean_velocity"][0], .1, delta=.002)
            self.assertLess(late["airborne"]["mean_displacement"][2], -.04)
        finally:
            gs.destroy()


if __name__=="__main__":unittest.main()
