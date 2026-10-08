"""Policy boundary tests only; no network, simulated success, or API charges."""
from types import SimpleNamespace
from unittest.mock import Mock
import tomllib

import pytest
from PIL import Image
from pydantic import ValidationError

from dressing_lab import ROOT, Action, ModelProposal, Observation, decide


def observation(tmp_path):
    Image.new("RGB", (16, 16), "white").save(tmp_path / "view.png")
    return Observation(rgb={"overhead": "view.png", "wrist": "view.png"}, robot={
        "left_position_m": [.3, 0, .5], "right_position_m": [.3, .5, .5],
        "left_gripper": 1, "right_gripper": 1})


def action(**kwargs):
    return Action(**({"left_delta_m": [0, 0, .01], "right_delta_m": [0, 0, 0],
                     "left_gripper": 1, "right_gripper": 1} | kwargs))


def test_reject_hidden_state_and_nan(tmp_path):
    data = observation(tmp_path).model_dump()
    with pytest.raises(ValidationError):
        Observation.model_validate(data | {"cloth_nodes": [[1, 2, 3]]})
    with pytest.raises(ValidationError):
        action(left_delta_m=[float("nan"), 0, 0])


def test_limit_is_vector_norm():
    with pytest.raises(ValueError):
        action(left_delta_m=[.019, .019, 0]).check_limits(.02)


def test_before_images_require_action(tmp_path):
    data = observation(tmp_path).model_dump()
    with pytest.raises(ValidationError):
        Observation.model_validate(data | {"previous_rgb": data["rgb"]})


def test_side_camera_requires_matching_history(tmp_path):
    data=observation(tmp_path).model_dump()
    side=data["rgb"]|{"table_side":"view.png"}
    assert set(Observation.model_validate(data|{"rgb":side}).rgb)=={"overhead","wrist","table_side"}
    with pytest.raises(ValidationError):
        Observation.model_validate(data|{"rgb":side,"previous_rgb":data["rgb"],"previous_action":action()})


def test_real_image_payload_and_refusal(tmp_path):
    client = Mock()
    config = tomllib.loads((ROOT / "config.toml").read_text())
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=action())
    result, _ = decide(client, "test-only", "a", observation(tmp_path), config, tmp_path)
    assert result.left_delta_m == [0, 0, .01]
    payload = client.responses.parse.call_args.kwargs
    assert payload["store"] is False
    assert sum(p["type"] == "input_image" for p in payload["input"][1]["content"]) == 2
    assert "view.png" not in str(payload)
    client.responses.parse.return_value = SimpleNamespace(status="incomplete", output_parsed=None)
    with pytest.raises(RuntimeError):
        decide(client, "test-only", "a", observation(tmp_path), config, tmp_path)


def test_model_proposal_not_an_action(tmp_path):
    kwargs = dict(garment_young_pa_low=1000, garment_young_pa_high=10000,
                  contact_friction_low=.1, contact_friction_high=.8,
                  left_target_m=[.3, 0, .5], right_target_m=[.3, .5, .5],
                  evidence="No interaction evidence; unvalidated prior.")
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=ModelProposal(**kwargs))
    config = tomllib.loads((ROOT / "config.toml").read_text())
    result, _ = decide(client, "test-only", "b", observation(tmp_path), config, tmp_path)
    assert isinstance(result, ModelProposal)
    with pytest.raises(ValidationError):
        ModelProposal(**(kwargs | {"contact_friction_low": 1}))


def test_scene_assets_and_initial_physics(tmp_path, monkeypatch):
    import mujoco
    import numpy as np
    from dressing_lab import build_scene
    monkeypatch.chdir(tmp_path)
    config = tomllib.loads((ROOT / "config.toml").read_text())
    model, data, report = build_scene(config, tmp_path)
    assert model.nu == 16
    assert list(model.flex_dim) == [3, 2]
    assert report["garment_simulation"]["components"] == 1
    assert report["garment_simulation"]["boundary_loops"] == 4
    assert model.ncam == 5
    side=model.camera("table_side").id
    assert abs(data.cam_xmat[side].reshape(3,3)[2,2])<1e-8
    assert data.cam_xpos[side,2]==pytest.approx(config["scene"]["table_height"]+.06)
    assert data.ncon == 0  # No initial robot/robot or table interpenetration.
    for side in ("left", "right"):
        assert model.site(side + "_grasp").id >= 0
        for j in range(1, 8):
            assert model.joint(f"{side}_joint{j}").id >= 0
    for i, name in enumerate(("sloth", "garment")):
        ids = model.flex_vertbodyid[model.flex_vertadr[i]:model.flex_vertadr[i]+model.flex_vertnum[i]]
        assert np.isclose(model.body_mass[ids].sum(), config["scene"][name]["mass"])
    mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert not any(w.number for w in data.warning)
    assert report["task_success"] is None


@pytest.fixture
def route_scene(tmp_path,monkeypatch):
    from dressing_lab import build_scene
    monkeypatch.chdir(tmp_path)
    config=tomllib.loads((ROOT/"config.toml").read_text())
    model,data,_=build_scene(config,tmp_path)
    return model,data,config


def route_action(**kwargs):
    from dressing_lab import RouteAAction
    return RouteAAction(**({"left_delta_m":[.01,0,0],"right_delta_m":[0,0,0],
        "left_rotation_rad":[0,0,0],"right_rotation_rad":[0,0,0],"left_gripper":1,"right_gripper":1}|kwargs))


def test_policy_packet_rejects_hidden_parameters(route_scene):
    from dressing_lab import policy_packet,PolicyPacket
    model,data,config=route_scene
    packet=policy_packet(model,data,"Move left gripper 1 cm",None,config)
    encoded=packet.model_dump_json()
    assert all(word not in encoded for word in ("young","poisson","mass","tetra","qpos","mesh","sloth"))
    bad=packet.model_dump();bad["camera_calibration"]["overhead"]["object_position"]=[0,0,0]
    with pytest.raises(ValidationError):PolicyPacket.model_validate(bad)
    with pytest.raises(ValidationError):PolicyPacket.model_validate(packet.model_dump()|{"material":8000})


def test_robot_ik_reaches_bounded_target_and_rejects_large_action(route_scene):
    import numpy as np
    import mujoco
    from dressing_lab import ik_controls
    model,data,config=route_scene
    original=data.qpos.copy()
    controls,targets=ik_controls(model,data,route_action(),config["control"])
    assert np.array_equal(data.qpos,original)  # Planning must not teleport simulator state.
    for side in ("left","right"):
        for j in range(1,8):
            data.qpos[model.joint(f"{side}_joint{j}").qposadr[0]]=controls[model.actuator(f"{side}_actuator{j}").id]
    mujoco.mj_kinematics(model,data)
    for side in ("left","right"):
        assert np.linalg.norm(data.site_xpos[model.site(side+"_grasp").id]-targets[side])<.001
    with pytest.raises(ValueError):ik_controls(model,data,route_action(left_delta_m=[.1,0,0]),config["control"])
    with pytest.raises(ValueError):route_action(left_rotation_rad=[.2,0,0]).check_control(config["control"])


def test_human_exit_is_idempotent_and_blocks_new_commands(tmp_path):
    from dressing_lab import ExperimentState
    state=ExperimentState(tmp_path,"test-model")
    state.submit("move left")
    state.finish("failure","user_exit")
    state.finish("success","second_request")
    assert state.outcome=="failure" and state.stop.is_set()
    with pytest.raises(ValueError):state.submit("keep moving")


def test_task_judgement_records_result_without_ending_experiment(tmp_path):
    import json
    from dressing_lab import ExperimentState
    state=ExperimentState(tmp_path,"test-model")
    with pytest.raises(ValueError):state.judge("success")
    first=state.submit("Lift the sloth")
    state.thread_id="same-experiment"
    revision=state.judge("success")
    assert revision>first and not state.stop.is_set() and not state.closed.is_set()
    assert state.status()["awaiting_instruction"] and state.status()["task_outcome"]=="success"
    assert state.judge("success")==revision and len(state.task_results)==1
    with pytest.raises(ValueError):state.judge("failure")
    state.submit("Now lift the shirt")
    assert not state.awaiting_instruction and state.task_outcome=="unjudged"
    assert state.thread_id=="same-experiment" and state.outcome=="unjudged"
    assert json.loads((tmp_path/"task_results.json").read_text())[0]["instruction"]=="Lift the sloth"
    state.finish()
    with pytest.raises(ValueError):state.judge("success")


def test_decision_schema_distinguishes_wait_and_action():
    from dressing_lab import RouteADecision
    with pytest.raises(ValidationError):RouteADecision(status="wait",actions=[route_action()]*5,message="")
    with pytest.raises(ValidationError):RouteADecision(status="act",actions=None,message="")
    for count in (0,1,4,6):
        with pytest.raises(ValidationError):RouteADecision(status="act",actions=[route_action()]*count,message="")
    assert len(RouteADecision(status="act",actions=[route_action()]*5,message="").actions)==5
    assert RouteADecision(status="wait",actions=None,message="").actions is None


def test_experiment_thread_survives_cancel_and_rejects_replacement(tmp_path):
    import json
    from dressing_lab import CleanPolicy
    policy=CleanPolicy(tmp_path,{})
    event=tmp_path/'events.jsonl'
    event.write_text(json.dumps({'type':'thread.started','thread_id':'experiment-one'})+'\n')
    policy.observe_thread(event)
    policy.cancel()
    policy.observe_thread(event)
    assert policy.thread_id=='experiment-one'
    event.write_text(json.dumps({'type':'thread.started','thread_id':'unexpected-new-thread'})+'\n')
    with pytest.raises(RuntimeError,match='Conversation changed'):policy.observe_thread(event)
    other=CleanPolicy(tmp_path/'other-experiment',{})
    assert other.thread_id is None and other.home!=policy.home
    assert not list(other.home.iterdir())


def test_execution_feedback_is_allowlisted(route_scene):
    from dressing_lab import policy_packet,PolicyPacket
    model,data,config=route_scene
    packet=policy_packet(model,data,'test',None,config).model_dump()
    for status in ('no_previous_decision','executed','partially_executed','not_executed'):
        assert PolicyPacket.model_validate(packet|{'execution_feedback':status}).execution_feedback==status
    with pytest.raises(ValidationError):
        PolicyPacket.model_validate(packet|{'execution_feedback':{'object_contact':True}})


def test_route_a_reports_namespace_permission_failure(monkeypatch):
    import dressing_lab
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(dressing_lab.subprocess, "run", Mock(return_value=SimpleNamespace(
        returncode=1, stderr="bwrap: setting up uid map: Permission denied\n")))
    with pytest.raises(RuntimeError, match="system/apparmor.d/dressing-bwrap"):
        dressing_lab.check_policy_isolation()


def test_route_a_checks_isolation_before_initializing_scene(monkeypatch):
    import dressing_lab
    scene = Mock()
    monkeypatch.setattr(dressing_lab, "scene_command", scene)
    monkeypatch.setattr(dressing_lab, "check_policy_isolation",
                        Mock(side_effect=RuntimeError("namespace permission denied")))
    monkeypatch.setattr(dressing_lab.sys, "argv", ["dressing", "route-a", "--open-browser"])
    with pytest.raises(SystemExit) as error:
        dressing_lab.main()
    assert error.value.code == 2
    scene.assert_not_called()


def test_genesis_cli_uses_same_route_a_without_initializing_warp(monkeypatch):
    import dressing_lab
    genesis=Mock();mujoco=Mock()
    monkeypatch.setattr(dressing_lab,"genesis_route_command",genesis)
    monkeypatch.setattr(dressing_lab,"scene_command",mujoco)
    monkeypatch.setattr(dressing_lab,"check_policy_isolation",Mock())
    monkeypatch.setattr(dressing_lab.sys,"argv",["dressing","route-a","--simulator","genesis","--genesis-device","cpu"])
    dressing_lab.main()
    genesis.assert_called_once();mujoco.assert_not_called()
    assert genesis.call_args.args[1].genesis_device=="cpu"


def test_backend_observation_allowlist_rejects_particle_and_material_fields():
    from dressing_lab import make_policy_packet,PoseState,PolicyPacket,DISPLAY_CAMERAS
    config=tomllib.loads((ROOT/"config.toml").read_text())
    pose=PoseState(left_position_m=[0,0,1],right_position_m=[.1,0,1],
        left_quaternion_wxyz=[1,0,0,0],right_quaternion_wxyz=[1,0,0,0],left_gripper=1,right_gripper=1)
    camera={"position_world_m":[0,0,2],"rotation_camera_to_world":[[1,0,0],[0,1,0],[0,0,1]],
            "vertical_fov_deg":43,"image_width":640,"image_height":480,"axes":"camera +X right, +Y up, viewing direction -Z"}
    calibration={name:camera for name in DISPLAY_CAMERAS}
    packet=make_policy_packet(pose,calibration,"move 1 cm",None,config)
    assert set(packet.model_dump())=={"task","robot","previous_action","previous_images_available",
        "coordinate_system","camera_calibration","action_limits","execution_feedback","chunk_execution"}
    with pytest.raises(ValidationError):
        make_policy_packet(pose,calibration|{"overhead":camera|{"particles":[[0,0,1]]}},"test",None,config)
    with pytest.raises(ValidationError):
        make_policy_packet(pose.model_dump()|{"material":8000},calibration,"test",None,config)
    with pytest.raises(ValidationError):
        PolicyPacket.model_validate(packet.model_dump()|{"chunk_execution":[{
            "index":1,"executed":True,"completed":False,"executed_simulation_seconds":.2,"object_contact":True}]})


@pytest.fixture
def visual_loop_scene():
    import numpy as np
    import dressing_lab
    config=tomllib.loads((ROOT/"config.toml").read_text())
    config["route_a"].update(camera_width=16,camera_height=16,video_fps=5)
    class TestScene:
        simulator="genesis";device="cpu";time=2.
        limitations=["Synthetic controller fixture; no physics or real model call"]
        def __init__(self,timestep=.01):
            self.timestep=timestep;self.values=np.array([0.,0.,1.,.1,0.,1.])
            self.control_history=[];self.step_history=[];self.plans=[]
            self.reject_at=None;self.on_reject=None;self.on_step=None;self.holds=0
        def pose(self):
            return dressing_lab.PoseState(left_position_m=self.values[:3].tolist(),right_position_m=self.values[3:].tolist(),
                left_quaternion_wxyz=[1,0,0,0],right_quaternion_wxyz=[1,0,0,0],left_gripper=1,right_gripper=1)
        def capture(self):return {name:np.full((16,16,3),100,dtype=np.uint8) for name in dressing_lab.DISPLAY_CAMERAS}
        def controls(self):return self.values.copy()
        def plan(self,action):
            action.check_control(config["control"])
            self.plans.append((action,self.values.copy()))
            if len(self.plans)==self.reject_at:
                if self.on_reject:self.on_reject()
                raise ValueError("Fixture IK rejection")
            target=self.values+np.array(action.left_delta_m+action.right_delta_m)
            return target,{side:target[start:start+3].tolist() for side,start in (("left",0),("right",3))}
        def set_control(self,values):
            self.values=values.copy();self.control_history.append(values.copy())
        def step(self,count):
            self.time+=count*self.timestep;self.step_history.append(count)
            if self.on_step:self.on_step()
        def hold(self):self.holds+=1
        def packet(self,task,previous_action):
            camera={"position_world_m":[0,0,2],"rotation_camera_to_world":[[1,0,0],[0,1,0],[0,0,1]],
                "vertical_fov_deg":43,"image_width":16,"image_height":16,"axes":"camera +X right, +Y up, viewing direction -Z"}
            return dressing_lab.make_policy_packet(self.pose(),{name:camera for name in dressing_lab.DISPLAY_CAMERAS},task,previous_action,config)
    return TestScene,config


@pytest.mark.parametrize("timestep", [.001, .005, .01])
@pytest.mark.parametrize("include_side", [False,True])
def test_common_visual_loop_preserves_conversation_feedback_and_videos(tmp_path,monkeypatch,timestep,visual_loop_scene,include_side):
    import json
    import numpy as np
    import dressing_lab
    TestScene,config=visual_loop_scene
    config["observation"]["cameras"]=list(dressing_lab.DISPLAY_CAMERAS if include_side else ("overhead","wrist"))
    calls=[]
    class TestPolicy:
        thread_id="fixture-one-experiment"
        def __init__(self,*args):self.memory=None
        def cancel(self):pass
        def decide(self,packet,images,previous_images,step_dir,cancelled):
            assert set(images)==set(packet.camera_calibration)==set(config["observation"]["cameras"])
            if previous_images is not None:assert set(previous_images)==set(images)
            calls.append((packet.execution_feedback,previous_images is not None))
            if self.memory is None:
                self.memory=packet.task.rsplit("：",1)[-1];message="fixture wait"
            else:message=self.memory
            return dressing_lab.RouteADecision(status="wait",actions=None,message=message),self.thread_id
    monkeypatch.setattr(dressing_lab,"CleanPolicy",TestPolicy)
    server=SimpleNamespace(shutdown=Mock(),server_close=Mock())
    def serve(state,port):state.url="http://127.0.0.1:0";return server
    monkeypatch.setattr(dressing_lab,"serve_dashboard",serve)
    scene=TestScene(timestep)
    dressing_lab.run_route_a(config,SimpleNamespace(smoke_test=True,open_browser=False),tmp_path,engine=scene)
    session=json.loads((tmp_path/"session.json").read_text())
    assert calls==[("no_previous_decision",False),("not_executed",True)]
    assert session["thread_id"]=="fixture-one-experiment" and session["executed_actions"]==2
    assert session["simulator"]=="genesis" and session["device"]=="cpu"
    assert session["task_success"] is None and session["error"] is None
    # Each one-second move must ramp over 100 target updates for either backend.
    assert len(scene.control_history)==200
    assert 0<scene.control_history[0][0]<.01
    assert np.all(np.diff([values[0] for values in scene.control_history[:100]])>0)
    assert scene.control_history[99][0]==pytest.approx(.01)
    assert scene.control_history[-1][0]==pytest.approx(0)
    assert max(scene.step_history)*timestep==pytest.approx(.01)
    assert scene.time==pytest.approx(4.)
    assert all((tmp_path/name).stat().st_size>0 for name in ("overhead.mp4","wrist.mp4","table_side.mp4","dual.mp4","triple.mp4"))


@pytest.mark.parametrize("scenario", ["complete", "reject", "stop", "revision"])
def test_five_chunks_execute_sequentially_and_cancel_remaining(tmp_path,monkeypatch,visual_loop_scene,scenario):
    import json
    import numpy as np
    import dressing_lab as lab
    TestScene,config=visual_loop_scene
    scene=TestScene();states=[];packets=[]
    actions=[route_action(left_delta_m=[dx,0,0]) for dx in (.01,.005,-.01,-.005,.01)]
    class Policy:
        thread_id="fixture-batch-experiment"
        def __init__(self,*args):pass
        def cancel(self):pass
        def decide(self,packet,images,previous_images,step_dir,cancelled):
            packets.append(packet)
            if len(packets)==1:
                answer=lab.RouteADecision(status="act",actions=actions,message="fixture only")
                lab.atomic_json(step_dir/"decision.json",{"decision":answer.model_dump(),"executed":False})
                return answer,self.thread_id
            states[0].finish(reason="fixture_complete");states[0].closed.set()
            return lab.RouteADecision(status="wait",actions=None,message="fixture done"),self.thread_id
    def serve(state,port):
        states.append(state);state.url="offline-fixture";state.submit("Move in five chunks")
        return SimpleNamespace(shutdown=Mock(),server_close=Mock())
    def interrupt():
        if len(scene.step_history)==20:
            if scenario=="stop":
                states[0].finish(reason="fixture_stop");states[0].closed.set()
            elif scenario=="revision":states[0].submit("New instruction: hold position")
    scene.on_step=interrupt
    if scenario=="reject":
        scene.reject_at=3
        scene.on_reject=lambda:states[0].submit("New instruction after rejected chunk")
    monkeypatch.setattr(lab,"CleanPolicy",Policy);monkeypatch.setattr(lab,"serve_dashboard",serve)
    lab.run_route_a(config,SimpleNamespace(smoke_test=False,open_browser=False),tmp_path,engine=scene)
    session=json.loads((tmp_path/"session.json").read_text())
    batch=json.loads((tmp_path/"steps/000001/execution.json").read_text())
    decision=json.loads((tmp_path/"steps/000001/decision.json").read_text())
    assert session["error"] is None and session["saved"]
    assert session["action_chunks_per_decision"]==5 and session["duration_seconds_per_chunk"]==1
    assert states[0].status()["action_chunks_per_decision"]==5
    assert states[0].status()["duration_seconds_per_chunk"]==1
    assert batch["executed"] is True and decision["executed_chunks"]==batch["executed_chunks"]
    if scenario=="complete":
        assert len(scene.plans)==5 and len(packets)==2
        assert batch["completed"] and batch["completed_chunks"]==5
        assert scene.time==pytest.approx(7.)
        assert [item["executed_simulation_seconds"] for item in batch["chunks"]]==[1.]*5
        for i,(_,pose_before) in enumerate(scene.plans):
            assert pose_before[0]==pytest.approx(sum(action.left_delta_m[0] for action in actions[:i]))
        assert packets[1].execution_feedback=="executed"
        assert len(packets[1].chunk_execution)==5
    elif scenario=="reject":
        assert len(scene.plans)==3 and len(scene.control_history)==200
        assert batch["completed_chunks"]==2 and not batch["completed"]
        assert batch["interrupted_reason"]=="validation_rejected"
        assert packets[1].execution_feedback=="partially_executed"
        assert [item.completed for item in packets[1].chunk_execution]==[True,True,False]
    else:
        assert len(scene.plans)==1 and len(scene.control_history)==20
        assert batch["executed_chunks"]==1 and batch["completed_chunks"]==0
        assert batch["chunks"][0]["executed_simulation_seconds"]==pytest.approx(.2)
        assert batch["interrupted_reason"]==("user_exit" if scenario=="stop" else "new_instruction")
        assert scene.holds>=2
        if scenario=="revision":
            assert len(packets)==2 and packets[1].execution_feedback=="partially_executed"
            assert packets[1].chunk_execution[0].executed_simulation_seconds==pytest.approx(.2)
        else:assert len(packets)==1


@pytest.mark.parametrize("judge_at", ["thinking", "executing", "completed"])
def test_task_judgement_pauses_and_next_task_keeps_scene_and_conversation(tmp_path,monkeypatch,visual_loop_scene,judge_at):
    import json
    import numpy as np
    import dressing_lab as lab
    TestScene,config=visual_loop_scene
    config["route_a"]["max_steps"]=5
    scene=TestScene();states=[];packets=[];judged_poses=[];holds_at_judgement=[];policies=[]
    first_steps={"thinking":0,"executing":20,"completed":500}[judge_at]

    def judge_current():
        judged_poses.append(scene.values.copy());holds_at_judgement.append(scene.holds)
        states[0].judge("success")
        assert not states[0].stop.is_set() and not states[0].closed.is_set()

    class Policy:
        thread_id="same-multi-task-experiment"
        def __init__(self,*args):policies.append(self);self.memory=None
        def cancel(self):pass
        def decide(self,packet,images,previous_images,step_dir,cancelled):
            packets.append(packet)
            if self.memory is None:
                self.memory="Lift the sloth"
                if judge_at=="thinking":judge_current()
                action=route_action(left_delta_m=[.001,0,0])
            else:
                assert self.memory=="Lift the sloth"
                assert "Now lift the shirt" in packet.task
                assert "Human result for user instruction 1: success" in packet.task
                np.testing.assert_allclose(scene.values,judged_poses[0])
                action=route_action(left_delta_m=[0,0,0],right_delta_m=[.002,0,0])
            answer=lab.RouteADecision(status="act",actions=[action]*5,message="fixture only")
            lab.atomic_json(step_dir/"decision.json",{"decision":answer.model_dump(),"executed":False})
            return answer,self.thread_id

    def serve(state,port):
        states.append(state);state.url="offline-multi-task-fixture";state.submit("Lift the sloth")
        return SimpleNamespace(shutdown=Mock(),server_close=Mock())

    def on_step():
        count=len(scene.step_history)
        if first_steps and count==first_steps and not states[0].task_results:judge_current()
        elif count==first_steps+500 and len(states[0].task_results)==1:judge_current()

    original_capture=scene.capture
    def capture():
        if states and states[0].awaiting_instruction and states[0].task_results:
            stage=len(states[0].task_results)
            if scene.holds>holds_at_judgement[-1]:
                assert not states[0].stop.is_set()
                np.testing.assert_allclose(scene.values,judged_poses[-1])
                if stage==1:states[0].submit("Now lift the shirt")
                else:
                    states[0].finish(reason="fixture_multi_task_complete");states[0].closed.set()
        return original_capture()

    scene.on_step=on_step;scene.capture=capture
    monkeypatch.setattr(lab,"CleanPolicy",Policy);monkeypatch.setattr(lab,"serve_dashboard",serve)
    lab.run_route_a(config,SimpleNamespace(smoke_test=False,open_browser=False),tmp_path,engine=scene)
    session=json.loads((tmp_path/"session.json").read_text())
    results=json.loads((tmp_path/"task_results.json").read_text())
    assert len(policies)==1 and len(packets)==2
    assert session["thread_id"]=="same-multi-task-experiment" and session["error"] is None
    assert session["task_success"] is None and session["human_outcome"]=="unjudged"
    assert session["task_results"]==results and [r["instruction_index"] for r in results]==[1,2]
    assert all(r["human_outcome"]=="success" for r in results)
    assert scene.time==pytest.approx(2+first_steps*.01+5)
    assert scene.values[0]==pytest.approx(judged_poses[0][0])
    assert scene.values[3]==pytest.approx(.11)
    if judge_at=="thinking":
        assert len(scene.plans)==5 and packets[1].previous_action is None
        assert packets[1].execution_feedback=="not_executed"
    else:
        batch=json.loads((tmp_path/"steps/000001/execution.json").read_text())
        assert packets[1].previous_action is not None
        if judge_at=="executing":
            assert batch["interrupted_reason"]=="human_judgement" and len(scene.plans)==6
            assert packets[1].execution_feedback=="partially_executed"
            assert batch["chunks"][0]["executed_simulation_seconds"]==pytest.approx(.2)
        else:
            assert len(scene.plans)==10 and packets[1].execution_feedback=="executed"
    assert all((tmp_path/name).stat().st_size>0 for name in ("triple.mp4","dual.mp4","overhead.mp4","wrist.mp4","table_side.mp4"))


def test_every_chunk_is_checked_before_execution():
    import dressing_lab as lab
    config=tomllib.loads((ROOT/"config.toml").read_text())
    answer=lab.RouteADecision(status="act",actions=[route_action()]*4+[route_action(left_delta_m=[.1,0,0])],message="")
    with pytest.raises(ValueError,match="translation norm"):
        answer.check_control(config["control"])


def test_dashboard_serves_side_view_and_exact_video_paths(tmp_path,monkeypatch):
    import io
    import numpy as np
    import http.server
    import dressing_lab as lab
    handlers=[]
    def server(address,handler):
        handlers.append(handler)
        return SimpleNamespace(server_port=0,serve_forever=lambda:None)
    monkeypatch.setattr(http.server,"ThreadingHTTPServer",server)
    state=lab.ExperimentState(tmp_path,"test-model")
    state.update_frames({name:np.zeros((32,32,3),dtype=np.uint8) for name in lab.DISPLAY_CAMERAS},0.)
    lab.serve_dashboard(state,0)
    def get(path):
        handler=handlers[0].__new__(handlers[0]);handler.path=path;handler.wfile=io.BytesIO()
        codes=[];handler.send_response=codes.append;handler.send_header=lambda *args:None;handler.end_headers=lambda:None
        handler.do_GET();return codes[0],handler.wfile.getvalue()
    assert get('/frame/table_side.jpg')[0]==200
    assert get('/frame/table_side.jpg')[1].startswith(b'\xff\xd8')
    assert 'id="table_side"' in get('/')[1].decode()
    (tmp_path/'table_side.mp4').write_bytes(b'side-video-fixture')
    (tmp_path/'triple.mp4').write_bytes(b'three-view-fixture')
    assert get('/video/table_side.mp4')[0]==404
    state.saved=True
    assert get('/video/table_side.mp4')==(200,b'side-video-fixture')
    assert get('/video/triple.mp4')==(200,b'three-view-fixture')
    assert get('/video/../config.toml')[0]==404
    assert get('/private/genesis/scene/report.json')[0]==404
