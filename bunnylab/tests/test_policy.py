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


def test_single_goal_human_judgement_and_restart(tmp_path):
    import json
    from dressing_lab import ExperimentState,atomic_json
    state=ExperimentState(tmp_path,"test-model")
    with pytest.raises(ValueError):state.judge("success")
    state.submit("Dress the sloth")
    with pytest.raises(ValueError):state.submit("Replace the goal")
    with pytest.raises(ValueError):state.restart()
    state.model_claim="completed";state.finish(reason="model_completed")
    assert state.outcome=="unjudged"
    atomic_json(tmp_path/"session.json",{"task_success":None,"model_claim":"completed"})
    state.saved=True;state.finalized=True
    state.judge("failure")
    assert state.stop.is_set() and not state.closed.is_set()
    assert state.judge("failure")==state.revision and len(state.task_results)==1
    with pytest.raises(ValueError):state.judge("success")
    saved=json.loads((tmp_path/"session.json").read_text())
    assert saved["task_success"] is False and saved["model_claim"]=="completed"
    state.restart()
    assert state.restart_requested and state.closed.is_set()


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
    monkeypatch.setattr(dressing_lab,"check_dashboard_port",Mock())
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
        "coordinate_system","camera_calibration","action_limits","execution_feedback","chunk_execution","controller_feedback"}
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


@pytest.mark.parametrize("scenario", ["complete", "reject", "stop"])
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
        scene.on_reject=None
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


@pytest.mark.parametrize("scenario", ["completed", "unable", "wait", "reject_retry", "errors", "budget", "stall", "deadline", "human", "deadline_executing", "human_executing", "schema_error"])
def test_autonomous_goal_termination_and_feedback(tmp_path,monkeypatch,visual_loop_scene,scenario):
    import json
    import dressing_lab as lab
    TestScene,config=visual_loop_scene
    scene=TestScene();states=[];packets=[]
    config["route_a"].update(max_steps=10,max_consecutive_failures=2,max_no_progress_rounds=2)
    if scenario=="reject_retry":scene.reject_at=1
    if scenario=="deadline":config["route_a"]["max_wall_seconds"]=.05
    def interrupt_action():
        if len(scene.step_history)==20:
            if scenario=="deadline_executing":states[0].task_started=lab.time.monotonic()-1801
            elif scenario=="human_executing":states[0].judge("failure")
    scene.on_step=interrupt_action
    class Policy:
        thread_id="one-autonomous-thread"
        def __init__(self,*args):pass
        def cancel(self):pass
        def decide(self,packet,images,previous_images,step_dir,cancelled):
            packets.append(packet)
            if scenario=="schema_error":raise lab.PolicyCallError("invalid_json_schema","模型输出格式错误",False)
            if scenario=="errors":raise TimeoutError("private internal path must not reach next packet")
            if scenario=="deadline":
                while not cancelled():lab.time.sleep(.005)
                raise InterruptedError()
            if scenario=="human":
                states[0].judge("failure")
                return lab.RouteADecision(status="act",actions=[route_action()]*5,message="stale"),self.thread_id
            status=scenario if scenario in ("completed","unable") else "act"
            if scenario=="wait":status="wait" if len(packets)==1 else "completed"
            if scenario=="reject_retry" and len(packets)==3:status="completed"
            action=route_action(left_delta_m=[0,0,0] if scenario=="stall" else [.001,0,0])
            answer=lab.RouteADecision(status=status,actions=[action]*5 if status=="act" else None,
                message="fixture",stage="dress",progress="visual evidence only")
            lab.atomic_json(step_dir/"decision.json",{"decision":answer.model_dump()})
            return answer,self.thread_id
    class Recorder:
        error=None
        def __init__(self,state,config):self.state=state
        def close(self):
            self.state.closed.set()
            return {"fixture":True}
    def serve(state,port):
        states.append(state);state.url="fixture";state.submit("Help the sloth put on the shirt")
        return SimpleNamespace(shutdown=Mock(),server_close=Mock())
    monkeypatch.setattr(lab,"CleanPolicy",Policy);monkeypatch.setattr(lab,"serve_dashboard",serve)
    monkeypatch.setattr(lab,"VideoRecorder",Recorder)
    if scenario=="schema_error":
        with pytest.raises(RuntimeError,match="模型输出格式错误"):
            lab.run_route_a(config,SimpleNamespace(smoke_test=False,open_browser=False),tmp_path,engine=scene)
    else:
        lab.run_route_a(config,SimpleNamespace(smoke_test=False,open_browser=False),tmp_path,engine=scene)
    record=json.loads((tmp_path/"session.json").read_text())
    expected={"completed":"model_completed","unable":"model_unable","wait":"model_completed",
        "reject_retry":"model_completed","errors":"consecutive_failures","budget":"action_budget_limit",
        "stall":"no_observable_change","deadline":"wall_time_limit","human":"human_judgement",
        "deadline_executing":"wall_time_limit","human_executing":"human_judgement","schema_error":"policy_configuration_error"}
    assert record["ended_by"]==expected[scenario]
    assert record["task_success"] is (False if scenario.startswith("human") else None)
    assert record["automatic_success"] is None and record["saved"]
    assert all(packet.task=="Help the sloth put on the shirt" for packet in packets)
    if scenario=="wait":
        assert len(packets)==2 and scene.time==pytest.approx(3.)
        assert packets[1].controller_feedback=="observation_wait"
    if scenario=="reject_retry":
        assert packets[1].controller_feedback=="ik_rejected"
        assert packets[1].execution_feedback=="not_executed"
        assert packets[2].execution_feedback=="executed" and len(scene.plans)==6
    if scenario=="errors":
        assert len(packets)==2 and packets[1].controller_feedback=="policy_error"
        assert "private internal" not in packets[1].model_dump_json()
    if scenario in ("deadline","human"):assert not scene.plans
    if scenario.endswith("_executing"):
        assert len(scene.plans)==1 and scene.time==pytest.approx(2.2)
        batch=json.loads((tmp_path/"steps/000001/execution.json").read_text())
        assert batch["executed_chunks"]==1 and batch["completed_chunks"]==0
    if scenario=="schema_error":
        assert len(packets)==1 and not scene.plans
        assert record["error"]=="模型输出格式错误" and "模型输出格式错误" in states[0].message
    if scenario=="budget":assert record["executed_actions"]==10


def test_cli_new_experiment_rebuilds_and_does_not_open_duplicate_tab(monkeypatch):
    import dressing_lab as lab
    calls=[]
    def runner(config,args):
        calls.append(args.open_browser)
        return len(calls)==1
    monkeypatch.setattr(lab,"genesis_route_command",runner)
    monkeypatch.setattr(lab,"check_policy_isolation",lambda:None)
    monkeypatch.setattr(lab,"check_dashboard_port",lambda port:None)
    monkeypatch.setattr(lab.sys,"argv",["dressing","route-a","--simulator","genesis","--open-browser"])
    lab.main()
    assert calls==[True,False]


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


def test_dashboard_refresh_does_not_terminate_task():
    from dressing_lab import DASHBOARD
    assert "pagehide" not in DASHBOARD and "sendBeacon" not in DASHBOARD
    assert "post('/restart'" in DASHBOARD and "post('/close'" in DASHBOARD


def test_terminal_policy_statuses_require_no_actions():
    from dressing_lab import RouteADecision
    for status in ("completed","unable"):
        assert RouteADecision(status=status,actions=None,message="evidence").status==status
        with pytest.raises(ValidationError):RouteADecision(status=status,actions=[route_action()]*5,message="")


def test_exported_policy_schema_requires_all_fields_recursively():
    from dressing_lab import RouteADecision
    schema=RouteADecision.output_schema()
    assert {'stage','progress'}<=set(schema['required'])
    def check(node):
        if isinstance(node,dict):
            assert 'default' not in node
            if node.get('type')=='object':
                assert set(node['required'])==set(node['properties'])
                assert node['additionalProperties'] is False
            for child in node.values():check(child)
        elif isinstance(node,list):
            for child in node:check(child)
    check(schema)
    actions=next(option for option in schema['properties']['actions']['anyOf'] if option['type']=='array')
    assert actions['minItems']==actions['maxItems']==5


@pytest.mark.parametrize('nested',[False,True])
def test_cli_schema_error_is_classified_as_non_retryable(tmp_path,nested):
    import json
    from dressing_lab import policy_process_error
    error={'error':{'code':'invalid_json_schema','message':"Missing 'stage'."},'status':400}
    event={'type':'turn.failed','error':{'message':json.dumps(error)}} if nested else {'type':'error',**error}
    path=tmp_path/'events.jsonl';path.write_text(json.dumps(event)+'\n')
    result=policy_process_error(path,1)
    assert result.code=='invalid_json_schema' and not result.retryable
    assert 'invalid_json_schema' in str(result)


def test_unknown_process_error_remains_retryable_without_echoing_server_text(tmp_path):
    from dressing_lab import policy_process_error
    path=tmp_path/'events.jsonl';path.write_text('{"type":"error","message":"private server detail"}\n')
    result=policy_process_error(path,1)
    assert result.retryable and 'private server detail' not in str(result)


def test_busy_dashboard_port_prevents_simulation_start(monkeypatch,capsys):
    import dressing_lab as lab
    engine=Mock()
    monkeypatch.setattr(lab,'check_policy_isolation',lambda:None)
    monkeypatch.setattr(lab,'check_dashboard_port',Mock(side_effect=lab.DashboardPortError(8765)))
    monkeypatch.setattr(lab,'genesis_route_command',engine)
    monkeypatch.setattr(lab.sys,'argv',['dressing','route-a','--simulator','genesis'])
    with pytest.raises(SystemExit) as error:lab.main()
    assert error.value.code==2
    engine.assert_not_called()
    message=capsys.readouterr().err
    assert '8765' in message and '保存并退出服务' in message and 'Traceback' not in message


def test_busy_dashboard_bind_race_has_clear_error(tmp_path,monkeypatch):
    import errno,http.server
    import dressing_lab as lab
    monkeypatch.setattr(http.server,'ThreadingHTTPServer',Mock(side_effect=OSError(errno.EADDRINUSE,'in use')))
    with pytest.raises(lab.DashboardPortError,match='8765'):
        lab.serve_dashboard(lab.ExperimentState(tmp_path,'test'),8765)


def test_dashboard_port_preflight_closes_socket(monkeypatch):
    import errno,socket
    import dressing_lab as lab
    sock=Mock()
    sock.__enter__=Mock(return_value=sock);sock.__exit__=Mock(return_value=False)
    sock.bind.side_effect=OSError(errno.EADDRINUSE,'in use')
    monkeypatch.setattr(socket,'socket',Mock(return_value=sock))
    with pytest.raises(lab.DashboardPortError):lab.check_dashboard_port(8765)
    sock.__exit__.assert_called_once()
    sock.bind.assert_called_once_with(('127.0.0.1',8765))
