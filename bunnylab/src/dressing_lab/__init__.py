"""One CLI for scene validation and bounded, image-based A/B decisions."""
import argparse
import base64
import importlib.metadata
import json
import math
import os
from pathlib import Path
import queue
import threading
import subprocess
import sys
import time
import tomllib
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
DISPLAY_CAMERAS=("overhead","wrist","table_side")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RobotState(Strict):
    left_position_m: list[float] = Field(min_length=3, max_length=3)
    right_position_m: list[float] = Field(min_length=3, max_length=3)
    left_gripper: float = Field(ge=0, le=1)
    right_gripper: float = Field(ge=0, le=1)


class Action(Strict):
    left_delta_m: list[float] = Field(min_length=3, max_length=3)
    right_delta_m: list[float] = Field(min_length=3, max_length=3)
    left_gripper: float = Field(ge=0, le=1)
    right_gripper: float = Field(ge=0, le=1)

    def check_limits(self, limit):
        if not math.isfinite(limit) or not 0 < limit <= .05:
            raise ValueError("Translation limit must be finite and in (0, 0.05] m")
        if any(math.sqrt(sum(v*v for v in xyz)) > limit + 1e-12
               for xyz in (self.left_delta_m, self.right_delta_m)):
            raise ValueError("Action exceeds translation norm limit; rejected, not clipped")
        return self


class Observation(Strict):
    rgb: dict[Literal["overhead", "wrist", "table_side"], str]
    robot: RobotState
    previous_action: Action | None = None
    previous_rgb: dict[Literal["overhead", "wrist", "table_side"], str] | None = None

    @model_validator(mode="after")
    def cameras(self):
        if set(self.rgb) not in ({"overhead","wrist"},set(DISPLAY_CAMERAS)):
            raise ValueError("Overhead and wrist RGB are required; table_side is optional")
        if self.previous_rgb is not None and set(self.previous_rgb) != set(self.rgb):
            raise ValueError("Previous observations require the same cameras")
        if (self.previous_rgb is None) != (self.previous_action is None):
            raise ValueError("Previous image pair and intervening action must be provided together")
        return self


class RouteAAction(Action):
    left_rotation_rad: list[float] = Field(min_length=3, max_length=3)
    right_rotation_rad: list[float] = Field(min_length=3, max_length=3)

    def check_control(self, control):
        self.check_limits(control["max_translation_m"])
        for rot in (self.left_rotation_rad, self.right_rotation_rad):
            if math.sqrt(sum(x*x for x in rot)) > control["max_rotation_rad"]+1e-12:
                raise ValueError("Rotation exceeds allowed norm")
        return self


class RouteADecision(Strict):
    status: Literal["act", "wait", "completed", "unable"]
    actions: list[RouteAAction] | None = Field(min_length=5,max_length=5)
    message: str = Field(max_length=800)
    stage: str = Field(default="", max_length=120)
    progress: str = Field(default="", max_length=800)

    @classmethod
    def output_schema(cls):
        """The service requires every object property, even local defaulted fields."""
        schema=cls.model_json_schema()
        def strict(node):
            if isinstance(node,dict):
                node.pop("default",None)
                if node.get("type")=="object":
                    node["required"]=list(node.get("properties",{}))
                    node["additionalProperties"]=False
                for value in node.values():strict(value)
            elif isinstance(node,list):
                for value in node:strict(value)
        strict(schema)
        return schema

    @model_validator(mode="after")
    def consistent(self):
        if (self.status == "act") != (self.actions is not None):
            raise ValueError("act requires exactly five actions; other statuses require null")
        return self

    def check_control(self,control):
        for action in self.actions or []:action.check_control(control)
        return self


class PoseState(RobotState):
    left_quaternion_wxyz: list[float] = Field(min_length=4, max_length=4)
    right_quaternion_wxyz: list[float] = Field(min_length=4, max_length=4)


class CameraCalibration(Strict):
    position_world_m: list[float] = Field(min_length=3,max_length=3)
    rotation_camera_to_world: list[list[float]] = Field(min_length=3,max_length=3)
    vertical_fov_deg: float = Field(gt=0,lt=180)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    axes: str

    @model_validator(mode="after")
    def matrix_shape(self):
        if any(len(row)!=3 for row in self.rotation_camera_to_world):raise ValueError("Camera rotation must be 3x3")
        return self


class PolicyLimits(Strict):
    max_translation_m: float = Field(gt=0,le=.05)
    max_rotation_rad: float = Field(gt=0,le=.25)
    duration_seconds: float = Field(gt=0,le=1)
    action_chunks_per_decision: Literal[5] = 5
    workspace_min: list[float] = Field(min_length=3,max_length=3)
    workspace_max: list[float] = Field(min_length=3,max_length=3)


class ChunkExecution(Strict):
    index: int = Field(ge=1,le=5)
    executed: bool
    completed: bool
    executed_simulation_seconds: float = Field(ge=0,le=1.000000001)


class PolicyPacket(Strict):
    """The complete allowed model input. No scene config or simulator object fields."""
    task: str
    robot: PoseState
    previous_action: RouteAAction | None
    previous_images_available: bool
    coordinate_system: str
    camera_calibration: dict[Literal["overhead","wrist","table_side"],CameraCalibration]
    action_limits: PolicyLimits
    execution_feedback: Literal["no_previous_decision", "executed", "partially_executed", "not_executed"] = "no_previous_decision"
    chunk_execution: list[ChunkExecution] = Field(default_factory=list,max_length=5)
    controller_feedback: Literal["none", "translation_limit", "rotation_limit", "workspace_limit", "ik_rejected", "joint_limit", "action_rejected", "policy_error", "observation_wait"] = "none"

    @model_validator(mode="after")
    def cameras_and_history(self):
        if set(self.camera_calibration) not in ({"overhead","wrist"},set(DISPLAY_CAMERAS)):
            raise ValueError("Overhead and wrist cameras are required; table_side is optional")
        if self.previous_images_available!=(self.previous_action is not None):raise ValueError("History requires action and images together")
        return self


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    temporary.replace(path)


def robot_pose(model, data):
    import mujoco
    import numpy as np
    result = {}
    for side in ("left", "right"):
        sid = model.site(side+"_grasp").id
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat,data.site_xmat[sid])
        result[side+"_position_m"] = data.site_xpos[sid].tolist()
        result[side+"_quaternion_wxyz"] = quat.tolist()
        opening = sum(float(data.qpos[model.joint(f"{side}_finger_joint{i}").qposadr[0]]) for i in (1,2))/.08
        result[side+"_gripper"] = float(np.clip(opening,0,1))
    return PoseState(**result)


def policy_packet(model, data, task, previous_action, config):
    # Deliberate allowlist: calibration is public camera/robot geometry only.
    calibration = {}
    for camera in DISPLAY_CAMERAS:
        cid = model.camera(camera).id
        calibration[camera] = {"position_world_m":data.cam_xpos[cid].tolist(),
            "rotation_camera_to_world":data.cam_xmat[cid].reshape(3,3).tolist(),
            "vertical_fov_deg":float(model.cam_fovy[cid]),
            "image_width":config["route_a"]["camera_width"],"image_height":config["route_a"]["camera_height"],
            "axes":"camera +X right, +Y up, viewing direction -Z"}
    return make_policy_packet(robot_pose(model,data),calibration,task,previous_action,config)


def make_policy_packet(robot, calibration, task, previous_action, config):
    """Shared input allowlist for either physics engine; no object state."""
    control = config["control"]
    camera_names=config["observation"]["cameras"]
    if set(camera_names) not in ({"overhead","wrist"},set(DISPLAY_CAMERAS)) or len(camera_names)!=len(set(camera_names)):
        raise ValueError("Policy cameras must be overhead, wrist, and optionally table_side")
    if not set(calibration)<=set(DISPLAY_CAMERAS):raise ValueError("Unapproved camera calibration")
    calibration={name:calibration[name] for name in camera_names}
    return PolicyPacket(task=task,robot=robot,previous_action=previous_action,
        previous_images_available=previous_action is not None,
        coordinate_system="World metres; origin at floor beneath table center. +X towards right robot base, +Y towards robot bases/back of table, +Z up. End-effector pose is the grasp site at the finger tips. Quaternion order w,x,y,z. Rotations are world-frame rotation vectors in radians. Gripper 0 closed, 1 open (80 mm total width).",
        camera_calibration=calibration,action_limits={k:control[k] for k in
            ("max_translation_m","max_rotation_rad","duration_seconds","workspace_min","workspace_max")})


def ik_controls(model, data, action, control):
    """Robot-only inverse kinematics; no object pose, mesh or material queries."""
    import mujoco
    import numpy as np
    action.check_control(control)
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    result = data.ctrl.copy()
    targets = {}
    for side in ("left", "right"):
        sid = model.site(side+"_grasp").id
        target = data.site_xpos[sid]+np.asarray(getattr(action,side+"_delta_m"))
        if np.any(target<control["workspace_min"]) or np.any(target>control["workspace_max"]):
            raise ValueError("Target outside declared robot workspace")
        rot = np.asarray(getattr(action,side+"_rotation_rad")); theta=np.linalg.norm(rot)
        transform=np.eye(3)
        if theta>1e-12:
            x,y,z=rot/theta; skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            transform+=math.sin(theta)*skew+(1-math.cos(theta))*(skew@skew)
        target_rotation=transform@data.site_xmat[sid].reshape(3,3)
        joints=[model.joint(f"{side}_joint{i}").id for i in range(1,8)]
        qids=model.jnt_qposadr[joints]; vids=model.jnt_dofadr[joints]
        original=data.qpos[qids].copy()
        for _ in range(100):
            mujoco.mj_kinematics(model,scratch); mujoco.mj_comPos(model,scratch)
            pos_error=target-scratch.site_xpos[sid]
            current=scratch.site_xmat[sid].reshape(3,3)
            rot_error=.5*sum((np.cross(current[:,i],target_rotation[:,i]) for i in range(3)))
            if np.linalg.norm(pos_error)<.0005 and np.linalg.norm(rot_error)<.005: break
            jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
            mujoco.mj_jacSite(model,scratch,jp,jr,sid)
            jac=np.vstack([jp[:,vids],jr[:,vids]*.3]); error=np.r_[pos_error,rot_error*.3]
            delta=jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*1e-5,error)
            scratch.qpos[qids]=np.clip(scratch.qpos[qids]+np.clip(delta,-.04,.04),
                model.jnt_range[joints,0]+.005,model.jnt_range[joints,1]-.005)
        else: raise ValueError("IK did not converge; action rejected")
        if np.max(np.abs(scratch.qpos[qids]-original))>control["max_joint_delta_rad"]:
            raise ValueError("IK joint displacement exceeds control limit")
        for i,q in enumerate(scratch.qpos[qids],1):result[model.actuator(f"{side}_actuator{i}").id]=q
        result[model.actuator(f"{side}_actuator8").id]=getattr(action,side+"_gripper")*255
        targets[side]=target.tolist()
    return result,targets


class ModelProposal(Strict):
    """Hypotheses, not calibrated truth. Backend mapping requires rabbit model."""
    garment_young_pa_low: float = Field(ge=100, le=1e8)
    garment_young_pa_high: float = Field(ge=100, le=1e8)
    contact_friction_low: float = Field(ge=.01, le=2)
    contact_friction_high: float = Field(ge=.01, le=2)
    left_target_m: list[float] = Field(min_length=3, max_length=3)
    right_target_m: list[float] = Field(min_length=3, max_length=3)
    evidence: str

    @model_validator(mode="after")
    def ordered(self):
        if self.garment_young_pa_low > self.garment_young_pa_high:
            raise ValueError("Young modulus range is reversed")
        if self.contact_friction_low > self.contact_friction_high:
            raise ValueError("Friction range is reversed")
        return self


def image_part(path):
    from PIL import Image
    import io
    # Decode then re-encode, so only the image (not arbitrary file bytes) is sent.
    with Image.open(path) as image:
        if image.width * image.height > 16_000_000:
            raise ValueError("Image exceeds 16 megapixels")
        image = image.convert("RGB")
        image.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}


def decide(client, model, route, observation, config, base):
    schema = Action if route == "a" else ModelProposal
    control = config["control"]
    prompt = (
        "You control a simulated bimanual robot for the specified dressing task. "
        "Use only supplied camera images and robot state. World XYZ is metres. "
        "Do not claim task success. Do not produce or execute code. "
        + (f"Output end-effector translations with each vector norm <= {control['max_translation_m']} m; "
           f"duration is {control['duration_seconds']} s; keep orientation fixed. "
           "Gripper 0 is closed, 1 is open. If uncertain, use zero motion and preserve grippers."
           if route == "a" else
           "Propose uncertain garment Young modulus and effective contact friction ranges, "
           "and short-term world-coordinate end-effector targets. These are hypotheses for "
           "a model validator and planner, not commands or calibrated parameters. "
           "Without before/after interaction evidence explicitly state that parameters "
           "are unvalidated priors. Never infer certainty from appearance alone.")
    )
    content = [{"type": "input_text", "text": json.dumps({
        "task": config["experiment"]["task"],
        "robot": observation.robot.model_dump(),
        "previous_action": (observation.previous_action.model_dump()
                            if observation.previous_action else None),
    })}]
    for stage, images in (("before", observation.previous_rgb), ("current", observation.rgb)):
        for camera, path in (images or {}).items():
            content.append({"type": "input_text", "text": f"{stage} {camera} camera"})
            content.append(image_part((base / path).resolve()))
    response = client.responses.parse(
        model=model, store=False, max_output_tokens=config["api"]["max_output_tokens"],
        input=[{"role": "system", "content": prompt}, {"role": "user", "content": content}],
        text_format=schema,
    )
    if response.status != "completed" or response.output_parsed is None:
        raise RuntimeError("API returned an incomplete/refused/unparsed decision; no action accepted")
    output = response.output_parsed
    if route == "a":
        output.check_limits(control["max_translation_m"])
    return output, response


def doctor(config):
    import mujoco
    report = {"python": sys.version.split()[0], "environment": sys.prefix,
              "packages": {p: importlib.metadata.version(p)
                           for p in ("mujoco", "numpy", "pillow", "openai", "pydantic")},
              "api_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
              "model_configured": bool(os.environ.get("OPENAI_MODEL") or config["api"]["model"])}
    for name, path in config["assets"].items():
        report[f"{name}_asset_present"] = bool(path and (ROOT / path).is_file())
    robot = ROOT / config["assets"]["robot"]
    if robot.is_file():
        m = mujoco.MjModel.from_xml_path(str(robot))
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_step(m, d)
        report["robot_compile_and_step"] = True
    report["dressing_scene_ready"] = False
    report["sloth_scene_command"] = "uv run dressing scene --view"
    report["route_b_planner_ready"] = False
    try:
        check_policy_isolation()
        report["route_a_isolation_ready"] = True
    except RuntimeError as error:
        report["route_a_isolation_ready"] = False
        report["route_a_isolation_error"] = str(error)
    report["next"] = "Validate sloth scene on CUDA with the scene command; closed-loop dressing and route B planning remain unimplemented."
    return report


def mesh_topology(points, faces):
    """Check connected garment and preserve its openings during simplification."""
    import numpy as np
    edges, counts = np.unique(np.sort(np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
        axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Garment contains non-manifold edges")
    def components(edges, vertices):
        adjacency = {int(i): set() for i in vertices}
        for a, b in edges:
            adjacency[int(a)].add(int(b)); adjacency[int(b)].add(int(a))
        remaining = set(adjacency)
        result = 0
        while remaining:
            stack = [remaining.pop()]; result += 1
            while stack:
                for j in adjacency[stack.pop()]:
                    if j in remaining:
                        remaining.remove(j); stack.append(j)
        return result
    boundary = edges[counts == 1]
    return {"vertices": len(points), "faces": len(faces),
            "components": components(edges, np.arange(len(points))),
            "boundary_loops": components(boundary, np.unique(boundary))}


def build_scene(config, run):
    """Generate MJCF from self-contained assets; no policy receives these data."""
    import copy
    import hashlib
    import xml.etree.ElementTree as ET
    import numpy as np
    import fast_simplification
    import mujoco
    sc = config["scene"]
    robot_path = ROOT / config["assets"]["robot"]
    root = ET.parse(robot_path).getroot()
    root.set("model", "Dual Franka / deformable sloth and T-shirt")
    root.find("compiler").set("meshdir", str(robot_path.parent / "assets"))
    root.find("option").attrib.update(timestep=str(sc["timestep"]), integrator="discrete", solver=sc["solver"],
        iterations=str(sc["iterations"]), ls_iterations="20", tolerance=str(sc["tolerance"]), cone="pyramidal", jacobian="sparse")
    ET.SubElement(root, "size", memory="256M")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(sc["width"]), offheight=str(sc["height"]))
    ET.SubElement(visual, "headlight", ambient="0.25 0.25 0.25", diffuse="0.3 0.3 0.3")
    ET.SubElement(visual, "quality", shadowsize="2048")
    root.remove(root.find("keyframe"))
    wb = root.find("worldbody")
    robot = copy.deepcopy(wb.find("body")); wb.clear()
    sections = {tag: copy.deepcopy(root.find(tag)) for tag in ("actuator", "tendon", "equality", "contact")}
    for tag in sections:
        root.find(tag).clear()
    def fmt(values):
        return " ".join(f"{float(v):.9g}" for v in np.asarray(values).ravel())
    for side, yaw in (("left", -0.35), ("right", math.pi + 0.35)):
        prefix = side + "_"
        body = copy.deepcopy(robot)
        body.set("pos", fmt(sc[side + "_base"]))
        body.set("quat", fmt([math.cos(yaw/2), 0, 0, math.sin(yaw/2)]))
        trees = [body]
        for tag, template in sections.items():
            for child in template:
                el = copy.deepcopy(child); root.find(tag).append(el); trees.append(el)
        for tree in trees:
            for el in tree.iter():
                for attr in ("name", "joint", "joint1", "joint2", "body1", "body2", "tendon"):
                    if attr in el.attrib:
                        el.set(attr, prefix + el.get(attr))
        wb.append(body)
        hand = body.find(f".//body[@name='{side}_hand']")
        ET.SubElement(hand, "site", name=side + "_grasp", pos="0 0 0.1034", size="0.005", rgba="0 0 0 0")
        if side == "left":
            ET.SubElement(hand, "camera", name="wrist", pos="0.06 0 0.02", xyaxes="1 0 0 0 -1 0", fovy="80")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="3 3 .1", rgba=".13 .17 .22 1")
    th = sc["table_height"]
    ts = sc["table_half_size"]
    ET.SubElement(wb, "geom", name="table", type="box", pos=fmt([0, 0, th-ts[2]]),
                  size=fmt(ts), rgba=".66 .53 .37 1", friction=".7 .005 .0001")
    for x in (-ts[0]+.07, ts[0]-.07):
        for y in (-ts[1]+.07, ts[1]-.07):
            ET.SubElement(wb, "geom", type="box", pos=fmt([x,y,(th-2*ts[2])/2]),
                          size=fmt([.035,.035,(th-2*ts[2])/2]), rgba=".2 .24 .28 1")
    for side in ("left", "right"):
        bp = sc[side+"_base"]
        ET.SubElement(wb, "geom", type="box", pos=fmt([bp[0],bp[1],(bp[2]+th)/2]),
                      size=fmt([.1,.1,(bp[2]-th)/2]), rgba=".23 .28 .32 1")
    ET.SubElement(wb, "light", pos="0 -1 3", dir="0 .3 -1", diffuse=".55 .55 .55")
    ET.SubElement(wb, "light", pos="1 1 2.5", dir="-.3 -.3 -1", diffuse=".25 .25 .25", castshadow="false")
    for name, camera in sc["cameras"].items():
        pos = np.array(camera["position"]); target = np.array(camera["target"])
        z = pos-target; z /= np.linalg.norm(z)
        up = np.array([0., 0., 1.]) if abs(z[2]) < .99 else np.array([0., 1., 0.])
        x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z,x)
        ET.SubElement(wb, "camera", name=name, pos=fmt(pos), xyaxes=fmt(np.r_[x,y]), fovy=str(camera["fovy"]))
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="sloth_tex", type="2d", file=str(ROOT/config["assets"]["sloth_texture"]))
    ET.SubElement(asset, "material", name="sloth_mat", texture="sloth_tex", specular=".05", shininess=".05")
    ET.SubElement(asset, "material", name="garment_mat", rgba=".12 .48 .26 1", specular=".1")
    with np.load(ROOT/config["assets"]["sloth"], allow_pickle=False) as a:
        sp, se, uv = a["points"].copy(), a["tetra"].copy(), a["texcoord"].copy()
    vertices, faces = [], []
    garment_path = ROOT/config["assets"]["garment"]
    for line in garment_path.read_text().splitlines():
        tokens = line.split()
        if tokens and tokens[0] == "v": vertices.append([float(v) for v in tokens[1:4]])
        if tokens and tokens[0] == "f":
            polygon = [int(t.split("/")[0])-1 for t in tokens[1:]]
            faces.extend([[polygon[0],polygon[i],polygon[i+1]] for i in range(1,len(polygon)-1)])
    gp = np.array(vertices); ge = np.array(faces)
    original_topology = mesh_topology(gp, ge)
    gp, ge = fast_simplification.simplify(gp, ge, target_count=sc["garment"]["target_faces"], agg=5.)
    topology = mesh_topology(gp,ge)
    if any(topology[k] != original_topology[k] for k in ("components", "boundary_loops")):
        raise ValueError(f"Simplification changed garment topology: {original_topology} -> {topology}")
    # Original shirt is flat in XY, with two separated fabric layers in Z.
    gp -= (gp.max(0)+gp.min(0))/2
    gp *= sc["garment"]["width"] / np.ptp(gp[:,1])
    gp = gp[:,[1,0,2]]; gp[:,0] *= -1
    for name, points, elements, dim in (("sloth",sp,se,3),("garment",gp,ge,2)):
        params = sc[name]
        points[:, :2] += params["position_xy"]
        points[:, 2] += th + params["radius"] + .004 - points[:,2].min()
        names = []
        for i, point in enumerate(points):
            bn = f"{name}_{i}"; names.append(bn)
            body = ET.SubElement(wb,"body",name=bn,pos=fmt(point))
            mass = params["mass"]/len(points)
            ET.SubElement(body,"inertial",pos="0 0 0",mass=str(mass),diaginertia=fmt([mass*1e-5]*3))
            for axis in np.eye(3):
                ET.SubElement(body,"joint",type="slide",axis=fmt(axis),damping="0.0001")
            # Vertex colliders provide robust table/robot support. Flex surfaces
            # still collide with the other deformable, using a separate bitmask.
            ET.SubElement(body,"geom",type="sphere",size=str(params["radius"]),
                mass="0",contype="4",conaffinity="1",group="5",rgba="0 0 0 0",
                friction=f"{params['friction']} .005 .0001",solref=".004 1",solimp=".95 .99 .001")
        attrs = dict(name=name, dim=str(dim), body=" ".join(names),
                     element=" ".join(map(str,elements.ravel())),
                     radius=str(params["radius"]), material=name+"_mat")
        if name == "sloth": attrs["texcoord"] = fmt(uv)
        deformable = root.find("deformable")
        if deformable is None: deformable = ET.SubElement(root,"deformable")
        flex = ET.SubElement(deformable,"flex",**attrs)
        ET.SubElement(flex,"contact",contype="2",conaffinity="2",condim="3",
                      friction=f"{params['friction']} .005 .0001",selfcollide="none",
                      solref=".01 1",solimp=".9 .99 .001")
        ET.SubElement(flex,"edge",damping="0")
        elasticity = dict(young=str(params["young"]),poisson=str(params["poisson"]),damping=str(params["damping"]))
        if dim == 2: elasticity.update(thickness=str(params["thickness"]),elastic2d="both")
        ET.SubElement(flex,"elasticity",**elasticity)
    xml = run/"scene.xml"
    ET.indent(root); ET.ElementTree(root).write(xml, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    for side in ("left", "right"):
        for j, angle in enumerate(sc["home"],1):
            data.qpos[model.joint(f"{side}_joint{j}").qposadr[0]] = angle
            data.ctrl[model.actuator(f"{side}_actuator{j}").id] = angle
        for j in (1,2): data.qpos[model.joint(f"{side}_finger_joint{j}").qposadr[0]] = .04
        data.ctrl[model.actuator(f"{side}_actuator8").id] = 255
    mujoco.mj_forward(model,data)
    np.savez(run/"initial_state.npz",qpos=data.qpos,ctrl=data.ctrl)
    metadata = {"garment_original":original_topology,"garment_simulation":topology,
                "sloth_vertices":len(sp),"sloth_tetrahedra":len(se),
                "asset_sha256":{k:hashlib.sha256((ROOT/config["assets"][k]).read_bytes()).hexdigest()
                                for k in ("robot","garment","sloth","sloth_texture")},
                "materials":{k:sc[k] for k in ("sloth","garment")},
                "limitations":["Material parameters are uncalibrated simulation priors.",
                    "Table/robot contact uses discrete vertex spheres, not a continuous fabric surface.",
                    "Flex self-collision disabled; garment layers can interpenetrate.",
                    "Scene validation is not grasping or dressing success."],
                "task_success":None,"policy_called":False}
    return model,data,metadata


def scene_command(config, args):
    sys.stdout.reconfigure(line_buffering=True)
    if not args.view:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import mujoco
    from PIL import Image
    run = ROOT/"runs"/(time.strftime("%Y%m%d-%H%M%S")+f"-{args.command}-"+uuid4().hex[:6])
    run.mkdir(parents=True)
    (run/"config.toml").write_text((ROOT/"config.toml").read_text())
    # MuJoCo's own diagnostics and all caches belong to this run.
    os.chdir(run)
    os.environ["XDG_CACHE_HOME"] = str(run/"cache")
    cache = ROOT/"runs"/"gpu-runtime"/"cache"
    os.environ["WARP_CACHE_PATH"] = str(cache/"warp")
    os.environ["CUDA_CACHE_PATH"] = str(cache/"cuda")
    print(f"Run: {run}",flush=True)
    model,data,report = build_scene(config,run)
    sc = config["scene"]
    report.update(backend=sc["backend"],mujoco=mujoco.__version__,nq=model.nq,nv=model.nv)
    if args.build_only:
        (run/"report.json").write_text(json.dumps(report,indent=2)); return
    if sc["backend"] != "warp" or not sc["device"].startswith("cuda:"):
        raise ValueError("This scene requires Warp on CUDA; CPU fallback is deliberately disabled")
    import warp as wp
    import mujoco_warp as mjw
    wp.config.kernel_cache_dir = str(cache/"warp")
    wp.config.quiet = True
    wp.init()
    device = wp.get_device(sc["device"])
    if not device.is_cuda: raise RuntimeError("CUDA GPU required")
    report["physics_device"] = str(device)
    report["gpu_name"] = device.name
    report["mujoco_warp"] = importlib.metadata.version("mujoco-warp")
    report["warp"] = wp.__version__
    @wp.kernel
    def retain_overflows(current: wp.array(dtype=wp.int32), total: wp.array(dtype=wp.int32)):
        total[0] = total[0] | current[0]
    start = time.monotonic()
    with wp.ScopedDevice(device):
        wm = mjw.put_model(model)
        wd = mjw.put_data(model,data,nworld=1,nconmax=8192,njmax=16384)
        overflow_history = wp.zeros(1,dtype=wp.int32)
        with wp.ScopedCapture() as capture:
            mjw.step(wm,wd)
            wp.launch(retain_overflows,dim=1,inputs=[wd.overflow,overflow_history])
        graph = capture.graph
    report["gpu_setup_seconds"] = time.monotonic()-start
    last_contacts = [0]
    physics_wall = [0.0]
    def step(count):
        started = time.monotonic()
        for _ in range(count): wp.capture_launch(graph)
        wp.synchronize()
        physics_wall[0] += time.monotonic()-started
        if np.any(overflow_history.numpy()):
            report["gpu_overflow"] = overflow_history.numpy().tolist()
            report["settling_validation_passed"] = False
            (run/"report.json").write_text(json.dumps(report,indent=2))
            raise RuntimeError(f"GPU solver or capacity limit reached: {overflow_history.numpy()}")
        # Only transfer display state, not the full constraint matrices every frame.
        data.qpos[:] = wd.qpos.numpy()[0]
        data.qvel[:] = wd.qvel.numpy()[0]
        data.time = float(wd.time.numpy()[0])
        last_contacts[0] = int(wd.nacon.numpy()[0])
        mujoco.mj_kinematics(model,data)
        mujoco.mj_comPos(model,data)
        mujoco.mj_flex(model,data)
        mujoco.mj_camlight(model,data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Nonfinite simulation state")
    # Render on NVIDIA EGL; viewer uses the desktop OpenGL context.
    if not args.view: os.environ.setdefault("MUJOCO_GL","egl")
    render_config=config["route_a"] if args.command=="route-a" else {"camera_height":sc["height"],"camera_width":sc["width"]}
    renderer = mujoco.Renderer(model,height=render_config["camera_height"],width=render_config["camera_width"])
    from OpenGL import GL
    report["render_device"] = GL.glGetString(GL.GL_RENDERER).decode()
    frames = []
    def render(label):
        for camera in (*sc["cameras"],"wrist"):
            renderer.update_scene(data,camera=camera)
            Image.fromarray(renderer.render()).save(run/f"{label}-{camera}.png")
    render("initial")
    start = time.monotonic()
    duration = args.seconds if args.seconds is not None else sc["validation_seconds"]
    nframes = max(1,math.ceil(duration*sc["fps"]))
    substeps = max(1,round(1/(sc["fps"]*sc["timestep"])))
    initial = data.flexvert_xpos.copy()
    trace = []
    for frame in range(nframes):
        step(substeps)
        renderer.update_scene(data,camera="overview")
        frames.append(Image.fromarray(renderer.render()).resize((800,600)))
        trace.append({"time":float(data.time),"max_speed":float(np.max(np.abs(data.qvel))),
                      "min_flex_z":float(data.flexvert_xpos[:,2].min()),"contacts":last_contacts[0]})
    report["simulation_seconds"] = float(data.time)
    report["validation_wall_seconds"] = time.monotonic()-start
    report["realtime_factor_including_render"] = data.time/report["validation_wall_seconds"]
    report["physics_wall_seconds"] = physics_wall[0]
    report["physics_realtime_factor"] = data.time/physics_wall[0]
    report["gpu_overflow"] = overflow_history.numpy().tolist()
    report["max_vertex_displacement_m"] = float(np.linalg.norm(data.flexvert_xpos-initial,axis=1).max())
    report["finite"] = bool(np.isfinite(data.qpos).all())
    report["warnings"] = {mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(data.warning) if w.number}
    report["min_flex_z"] = float(data.flexvert_xpos[:,2].min())
    report["minimum_z_during_validation"] = min(t["min_flex_z"] for t in trace)
    report["settled_max_speed"] = float(np.max(np.abs(data.qvel)))
    report["settling_validation_passed"] = bool(report["finite"] and not report["warnings"]
        and report["minimum_z_during_validation"] >= sc["table_height"]-.005
        and report["settled_max_speed"] < .1)
    render("settled")
    frames[0].save(run/"settling.webp",save_all=True,append_images=frames[1:],duration=round(1000/sc["fps"]),loop=0)
    np.savez(run/"settled_state.npz",qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl)
    (run/"trace.json").write_text(json.dumps(trace,indent=2))
    (run/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if not report["settling_validation_passed"]:
        renderer.close()
        raise RuntimeError(f"Settling validation failed; inspect {run / 'report.json'}")
    if args.command=="route-a":
        def set_control(values):
            data.ctrl[:]=values
            wp.copy(wd.ctrl,wp.array([values],dtype=wp.float32,device=device))
        try:return run_route_a(config,args,run,model,data,renderer,step,set_control)
        finally:renderer.close()
        return
    renderer.close()
    if args.view:
        import mujoco.viewer
        paused = [not args.play]
        if args.play:
            # Replay from the initial scene so the settling is visible in the window.
            mujoco.mj_resetData(model,data)
            with np.load(run/"initial_state.npz",allow_pickle=False) as initial_state:
                data.qpos[:] = initial_state["qpos"]
                data.ctrl[:] = initial_state["ctrl"]
            mujoco.mj_forward(model,data)
            with wp.ScopedDevice(device):
                wd = mjw.put_data(model,data,nworld=1,nconmax=8192,njmax=16384)
                overflow_history.zero_()
                with wp.ScopedCapture() as replay_capture:
                    mjw.step(wm,wd)
                    wp.launch(retain_overflows,dim=1,inputs=[wd.overflow,overflow_history])
                graph = replay_capture.graph
        def key(keycode):
            if keycode == 32: paused[0] = not paused[0]
        with mujoco.viewer.launch_passive(model,data,key_callback=key) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = [0,0,.85]
            viewer.cam.distance = 2.3; viewer.cam.azimuth = -55; viewer.cam.elevation = -32
            print(f"Viewer ready. SPACE: pause/run; mouse: orbit/zoom. Starts {'paused' if paused[0] else 'running'}.",flush=True)
            while viewer.is_running():
                tick = time.monotonic()
                if not paused[0]:
                    with viewer.lock(): step(substeps)
                viewer.sync()
                time.sleep(max(0,1/sc["fps"]-(time.monotonic()-tick)))


class ExperimentState:
    def __init__(self, run, model, simulator="mujoco", device="cuda", duration_seconds=1.):
        self.run=run;self.lock=threading.RLock();self.stop=threading.Event();self.closed=threading.Event()
        self.frames=None;self.jpg={};self.revision=0;self.commands=[]
        self.phase="preparing";self.message="正在准备场景";self.step=0;self.sim_time=0.
        self.outcome="unjudged";self.reason=None;self.saved=False;self.error=None
        self.model=model;self.thread_id=None;self.url=None;self.simulator=simulator;self.device=device
        self.duration_seconds=duration_seconds
        self.awaiting_instruction=True;self.task_outcome="unjudged";self.task_results=[]
        self.restart_requested=False;self.finalized=False
        self.task_started=None;self.task_ended=None;self.model_claim=None;self.model_message=""
        self.stage="";self.progress="";self.decision_count=0;self.budget_used=0
        self.failures=0;self.no_progress=0;self.last_error=None
        self.limits={"max_steps":600,"max_wall_seconds":1800}

    def submit(self, text):
        text=text.strip()
        if not text or len(text)>4000:raise ValueError("目标需为 1–4000 个字符")
        with self.lock:
            if self.stop.is_set() or self.commands:
                raise ValueError("一场实验只接受一个目标；请结束后创建新实验")
            self.commands.append(text);self.revision+=1
            self.task_started=time.monotonic()
            self.awaiting_instruction=False
            self.phase="queued";self.message="目标已确定，开始自主执行"
            atomic_json(self.run/"instructions.json",self.commands)
            return self.revision

    def judge(self, outcome):
        """Human result is independent of the policy's completion claim."""
        if outcome not in ("success","failure"):raise ValueError("请选择成功或失败")
        with self.lock:
            if not self.commands:raise ValueError("请先开始任务")
            if self.restart_requested:raise ValueError("正在创建新实验")
            if self.task_results:
                if self.outcome==outcome:return self.revision
                raise ValueError("人工结果已经记录")
            self.task_results.append({"instruction_index":1,"instruction":self.commands[0],
                "human_outcome":outcome,"task_success":outcome=="success",
                "executed_actions_total":self.step,"simulation_seconds":self.sim_time})
            atomic_json(self.run/"task_results.json",self.task_results)
            self.task_outcome=outcome;self.outcome=outcome
            self.finish(outcome,"human_judgement")
            session=self.run/"session.json"
            if session.exists():
                record=json.loads(session.read_text())
                record.update(human_outcome=outcome,task_success=outcome=="success",task_results=self.task_results)
                atomic_json(session,record)
            return self.revision

    def finish(self, outcome="unjudged", reason="user_exit"):
        if outcome not in ("success","failure","unjudged"):raise ValueError("Unknown human outcome")
        with self.lock:
            if not self.stop.is_set():
                self.outcome=outcome;self.reason=reason;self.phase="stopping"
                self.message="正在停止动作并保存视频…";self.stop.set()

    def restart(self):
        with self.lock:
            if not self.saved or not self.finalized:raise ValueError("请等待本场实验保存完成")
            self.restart_requested=True;self.closed.set()

    def status(self):
        with self.lock:
            return {"run_id":self.run.name,"phase":self.phase,"message":self.message,
                "step":self.step,"simulation_seconds":self.sim_time,"instruction":self.commands[0] if self.commands else "",
                "revision":self.revision,"model":self.model,"thread_id":self.thread_id,
                "awaiting_instruction":self.awaiting_instruction,"task_outcome":self.task_outcome,
                "judged_tasks":len(self.task_results),"model_claim":self.model_claim,"model_message":self.model_message,
                "stage":self.stage,"progress":self.progress,"decision_count":self.decision_count,
                "budget_used":self.budget_used,"max_steps":self.limits["max_steps"],
                "max_wall_seconds":self.limits["max_wall_seconds"],
                "wall_seconds":max(0.,(self.task_ended or time.monotonic())-self.task_started) if self.task_started else 0.,
                "consecutive_failures":self.failures,"no_progress_rounds":self.no_progress,
                "simulator":self.simulator,"device":self.device,
                "action_chunks_per_decision":5,"duration_seconds_per_chunk":self.duration_seconds,
                "ended":self.stop.is_set(),"saved":self.saved,"finalized":self.finalized,
                "outcome":self.outcome,"error":self.error,"last_error":self.last_error,"end_reason":self.reason}

    def update_frames(self, frames, sim_time):
        from PIL import Image
        import io
        jpeg={}
        for camera,frame in frames.items():
            buffer=io.BytesIO();Image.fromarray(frame).save(buffer,format="JPEG",quality=88)
            jpeg[camera]=buffer.getvalue()
        with self.lock:
            self.frames={k:v.copy() for k,v in frames.items()};self.jpg=jpeg;self.sim_time=float(sim_time)


class VideoRecorder:
    """One wall-clock timeline, synchronized camera and combined MP4 outputs."""
    def __init__(self,state,config):
        import shutil
        if not shutil.which("ffmpeg"):raise RuntimeError("ffmpeg is required for recording")
        self.state=state;self.finished=threading.Event();self.error=None;self.frames_written=0
        self.fps=config["video_fps"];w=config["camera_width"];h=config["camera_height"]
        self.log=(state.run/"encoder.log").open("w")
        cmd=["ffmpeg","-hide_banner","-loglevel","error","-n","-f","rawvideo","-pix_fmt","rgb24",
             "-s",f"{3*w}x{h}","-r",str(self.fps),"-i","pipe:0","-filter_complex",
             f"[0:v]split=5[a][b][c][d][triple];[a]crop={w}:{h}:0:0[overhead];"
             f"[b]crop={w}:{h}:{w}:0[wrist];[c]crop={w}:{h}:{2*w}:0[table_side];[d]crop={2*w}:{h}:0:0[dual]"]
        for name in (*DISPLAY_CAMERAS,"dual","triple"):
            cmd.extend(["-map",f"[{name}]","-an","-c:v","libx264","-preset","ultrafast","-crf","23",
                "-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",str(state.run/(name+".mp4"))])
        self.process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log,
                                      start_new_session=True)
        self.thread=threading.Thread(target=self._record,daemon=True);self.thread.start()

    def _record(self):
        import numpy as np
        from PIL import Image,ImageDraw
        start=time.monotonic()
        try:
            with (self.state.run/"video_timeline.jsonl").open("w") as timeline:
                while not self.finished.is_set() or self.frames_written==0:
                    with self.state.lock:
                        frames=self.state.frames;status=self.state.status()
                    if frames is None:
                        if self.finished.wait(.02):break
                        continue
                    combined=Image.fromarray(np.concatenate([frames[name] for name in DISPLAY_CAMERAS],axis=1))
                    draw=ImageDraw.Draw(combined);w=frames["overhead"].shape[1]
                    for i,name in enumerate(DISPLAY_CAMERAS):
                        x=i*w;name=name.upper()
                        draw.rectangle((x,0,x+w,23),fill=(12,20,30))
                        draw.text((x+8,6),f"{name} | {status['phase']} | step {status['step']} | sim {status['simulation_seconds']:.3f}s",fill="white")
                    self.process.stdin.write(combined.tobytes())
                    timeline.write(json.dumps({"frame":self.frames_written,"wall_seconds":time.monotonic()-start,
                        "simulation_seconds":status["simulation_seconds"],"phase":status["phase"],
                        "step":status["step"],"instruction_revision":status["revision"]})+"\n")
                    self.frames_written+=1
                    self.finished.wait(max(0,start+self.frames_written/self.fps-time.monotonic()))
        except Exception as error:self.error=str(error)
        finally:
            try:self.process.stdin.close()
            except OSError:pass

    def close(self):
        self.finished.set();self.thread.join(timeout=20)
        if self.thread.is_alive():
            self.process.kill();self.thread.join(timeout=2);self.error="Encoder thread did not stop"
        try:code=self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill();self.process.wait();code=-1
        self.log.close()
        if code or self.error:raise RuntimeError(self.error or f"Video encoder failed: {code}")
        return {"frames":self.frames_written,"fps":self.fps,"duration_seconds":self.frames_written/self.fps,
                "files":[*(name+".mp4" for name in DISPLAY_CAMERAS),"dual.mp4","triple.mp4"],"timebase":"wall clock; includes decision waits"}


DASHBOARD = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>路线 A · 自主穿衣实验</title><style>
*{box-sizing:border-box}body{margin:0;background:#101822;color:#eaf0f6;font:16px system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:24px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}h1{font-size:24px;margin:0}small,.muted{color:#a0b4c8}.badge{background:#203a4f;color:#85d3ff;border-radius:20px;padding:8px 14px}
.views{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.panel{background:#1b2837;border:1px solid #324354;border-radius:12px;overflow:hidden}.label{padding:12px 16px;display:flex;justify-content:space-between}.panel img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#080e15}
.controls{padding:20px;margin-top:18px}textarea{width:100%;resize:vertical;min-height:90px;background:#101b27;color:white;border:1px solid #526579;border-radius:8px;padding:12px;font:inherit}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:12px}button,select{border:0;border-radius:8px;font:inherit;padding:11px 16px}button{background:#60b7ee;color:#0a2436;cursor:pointer}button.stop{background:#d46b66;color:white}button:disabled{opacity:.45;cursor:default}select{background:#33465a;color:white}.status{min-height:28px;margin:14px 0;color:#a5d8f5}a{color:#93d9ff}#videos{display:none}.facts{font-size:13px;color:#9ab0c4;margin-top:14px;word-break:break-all}@media(max-width:800px){.views{grid-template-columns:1fr}main{padding:12px}}
</style><main><header><div><h1>路线 A · 自主穿衣实验</h1><small>一句目标，自主操作到结束；新实验重置场景与 GPT 会话</small></div><span class="badge" id="backend">三 RGB</span></header>
<section class="views"><div class="panel"><div class="label"><b>俯视相机</b><small>固定安装</small></div><img id="overhead" alt="俯视相机画面"></div><div class="panel"><div class="label"><b>腕部相机</b><small>随左手腕运动</small></div><img id="wrist" alt="腕部相机画面"></div><div class="panel"><div class="label"><b>桌面侧视相机</b><small>与桌面平行</small></div><img id="table_side" alt="桌面水平侧视画面"></div></section>
<section class="panel controls"><form id="form"><label for="task">本场实验目标</label><textarea id="task" placeholder="帮助树懒将衣服穿好" maxlength="4000" required>帮助树懒将衣服穿好</textarea><div class="row"><button id="send">开始自主任务</button><span class="muted">一次实验一个目标 · Ctrl + Enter 开始 · Esc 停止</span></div></form>
<div class="status" id="status">连接场景中…</div><p id="plan"></p><p class="muted" id="budget"></p>
<div class="row"><button type="button" class="stop" id="stop">停止并保存视频</button><button type="button" id="restart" disabled>新实验：全部归位</button><button type="button" id="close">保存并退出服务</button></div>
<div class="row"><span>人工最终判定：</span><button type="button" id="task-success" disabled>成功</button><button type="button" id="task-failure" disabled>失败</button><span class="muted">模型完成声明不等于任务成功；判定会结束当前任务</span></div>
<p class="muted" id="instruction"></p><div id="videos"><b>视频已保存：</b> <a href="/video/triple.mp4" download>三视角</a> · <a href="/video/dual.mp4" download>双视角</a> · <a href="/video/overhead.mp4" download>俯视</a> · <a href="/video/wrist.mp4" download>腕部</a> · <a href="/video/table_side.mp4" download>桌面侧视</a></div><p class="muted">刷新或关闭页面不会重置任务，后台会继续运行至结束或预算上限。结束后可查看录像并判定；新实验保留旧记录。</p><div class="facts" id="facts"></div></section></main>
<script>
let ended=false,judging=false,restarting=false,runId=null,quitting=false;const $=id=>document.getElementById(id);
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Dressing-Client':'route-a'},body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw Error(j.error);return j}
$('form').onsubmit=async e=>{e.preventDefault();$('send').disabled=true;try{await post('/command',{text:$('task').value})}catch(e){$('status').textContent=e.message}};
async function judge(outcome){if(judging||restarting)return;judging=true;try{await post('/judge',{outcome})}catch(e){$('status').textContent=e.message}finally{judging=false}}
$('task-success').onclick=()=>judge('success');$('task-failure').onclick=()=>judge('failure');
async function stop(){if(ended||restarting)return;try{await post('/stop',{})}catch(e){$('status').textContent=e.message}}
$('stop').onclick=stop;
$('restart').onclick=async()=>{restarting=true;$('restart').disabled=true;$('status').textContent='正在重新加载初始场景与新会话…';try{await post('/restart',{})}catch(e){restarting=false;$('status').textContent=e.message}};
$('close').onclick=async()=>{try{await post('/close',{});quitting=true;$('status').textContent='正在保存并退出；下次运行启动命令会创建全新实验'}catch(e){$('status').textContent=e.message}};
document.addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();stop()}if(e.ctrlKey&&e.key==='Enter'&&!$('send').disabled)$('form').requestSubmit()});
async function poll(){if(quitting)return;try{const s=await(await fetch('/status',{cache:'no-store'})).json();
if(runId&&runId!==s.run_id){restarting=false;$('task').value='帮助树懒将衣服穿好';}runId=s.run_id;
ended=s.ended;$('send').disabled=ended||!!s.instruction||restarting;$('task').disabled=$('send').disabled;$('stop').disabled=ended||restarting;
$('restart').disabled=!s.saved||!s.finalized||restarting;
const canJudge=!judging&&!restarting&&!!s.instruction&&s.outcome==='unjudged';$('task-success').disabled=!canJudge;$('task-failure').disabled=!canJudge;
$('status').textContent=restarting?'正在重建初始场景，请稍候…':s.message;
$('instruction').textContent=(s.instruction?'固定目标：'+s.instruction:'等待任务目标')+' · 人工结果：'+({unjudged:'未判定',success:'成功',failure:'失败'}[s.outcome]);
$('plan').textContent='模型阶段：'+(s.stage||'—')+' · '+(s.progress||'暂无进展记录')+(s.model_claim?' · 模型结束状态：'+s.model_claim:'')+' · '+s.model_message;
$('budget').textContent='动作预算 '+s.budget_used+'/'+s.max_steps+'（含短暂等待） · 任务用时 '+s.wall_seconds.toFixed(0)+'/'+s.max_wall_seconds+' 秒 · 决策 '+s.decision_count+' 轮 · 连续错误 '+s.consecutive_failures+' · 连续无可见变化 '+s.no_progress_rounds+' 轮';
$('facts').textContent='每轮 '+s.action_chunks_per_decision+' 个动作 × '+s.duration_seconds_per_chunk+' s · 已执行 '+s.step+' 个动作 · 仿真 '+s.simulation_seconds.toFixed(3)+' s · 模型 '+s.model+' · 实验 '+s.run_id+(s.thread_id?' · 会话 '+s.thread_id:' · 会话尚未开始');
$('videos').style.display=s.saved?'block':'none';$('backend').textContent=s.simulator+' · '+s.device+' · 三 RGB';
for(const c of ['overhead','wrist','table_side'])$(c).src='/frame/'+c+'.jpg?t='+Date.now();
}catch(e){$('status').textContent=restarting?'正在重建场景，连接恢复后可开始新任务…':'连接中断；任务可能仍在后台执行，请查看终端'}setTimeout(poll,500)}poll();
</script></html>"""


class DashboardPortError(RuntimeError):
    def __init__(self, port):
        super().__init__(
            f"网页端口 {port} 已被占用，未启动新的实验。"
            f"如果是之前启动的实验，可访问 http://127.0.0.1:{port}；"
            "需要重新开始时，先在旧网页点击“保存并退出服务”或在旧终端按 Ctrl-C，再运行启动命令。"
        )


def check_dashboard_port(port):
    """Fail before expensive simulation setup; the real bind still guards races."""
    import errno
    import socket
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:probe.bind(("127.0.0.1",port))
        except OSError as error:
            if error.errno==errno.EADDRINUSE:raise DashboardPortError(port) from None
            raise


def serve_dashboard(state, port):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,code,body,kind="application/json; charset=utf-8"):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(code);self.send_header("Content-Type",kind)
            self.send_header("Content-Length",str(len(body)));self.send_header("Cache-Control","no-store")
            self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            path=self.path.split("?",1)[0]
            if path=="/":return self.send(200,DASHBOARD.encode(),"text/html; charset=utf-8")
            if path=="/status":return self.send(200,state.status())
            if path in tuple("/frame/"+name+".jpg" for name in DISPLAY_CAMERAS):
                with state.lock:frame=state.jpg.get(path.split("/")[-1][:-4])
                return self.send(200,frame,"image/jpeg") if frame else self.send(503,{"error":"Preparing cameras"})
            if path in tuple("/video/"+name+".mp4" for name in (*DISPLAY_CAMERAS,"dual","triple")) and state.saved:
                # Exactly these outputs, never general file serving.
                return self.send(200,(state.run/path.split("/")[-1]).read_bytes(),"video/mp4")
            self.send(404,{"error":"Not found"})
        def do_POST(self):
            origin=self.headers.get("Origin")
            if (origin and origin!=state.url) or (not origin and self.headers.get("X-Dressing-Client")!="route-a"):
                return self.send(403,{"error":"Local dashboard requests only"})
            try:
                length=int(self.headers.get("Content-Length",0))
                if not 0<length<=20000:raise ValueError("Invalid request size")
                payload=json.loads(self.rfile.read(length))
                if self.path=="/command":return self.send(200,{"revision":state.submit(payload["text"])})
                if self.path=="/judge":return self.send(200,{"revision":state.judge(payload["outcome"])})
                if self.path=="/restart":
                    state.restart();return self.send(200,{"restarting":True})
                if self.path=="/stop":
                    state.finish(payload.get("outcome","unjudged"));return self.send(200,{"stopping":True})
                if self.path=="/close":
                    state.finish(payload.get("outcome","unjudged"),"tab_closed");state.closed.set()
                    return self.send(200,{"stopping":True})
                self.send(404,{"error":"Not found"})
            except (ValueError,KeyError,TypeError) as error:self.send(400,{"error":str(error)})
    try:server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    except OSError as error:
        import errno
        if error.errno==errno.EADDRINUSE:raise DashboardPortError(port) from None
        raise
    state.url=f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return server


POLICY_INSTRUCTIONS = """You are the independent Route A visual robot policy, not a coding assistant.
Only use the supplied task, overhead/wrist RGB images and table_side RGB when supplied, camera calibration, robot end-effector poses,
gripper openings, and previous action and images. You have no object ground truth or material parameters.
Human task results describe the user's judgement of specific prior instructions, not automatic completion of other tasks.
Use the history of THIS experiment, including observations, decisions and execution feedback.
Do not access files, tools, networks, shell, external memory, other conversations, or request extra viewpoints.
A proposed action is not evidence of execution; only execution_feedback and observed robot states are evidence.
execution_feedback refers to the most recent decision, including a rejected or cancelled one.
previous_action and before images refer to the last actually executed action, which may be older.
chunk_execution reports which chunks of the most recent decision actually ran and for how long.
Any chunks not listed in chunk_execution were not executed.
Return only the provided JSON schema. For status='act', output actions with exactly FIVE ordered action chunks.
Each chunk runs for action_limits.duration_seconds (currently 1 second); five chunks run before the next decision.
Each translation and rotation is incremental relative to the actual robot pose at the START of that chunk,
not the pose in this observation and not an absolute target. Gripper values are absolute opening targets.
There are no new model observations between chunks. Plan conservatively using the current images.
Later chunks are discarded if a chunk is rejected, the budget expires, or the user stops.
Translation and rotation norms must obey action_limits. Both arms must be specified; zero motion is valid.
You have ONE fixed goal for the entire experiment. Autonomously choose and revise your own plan.
Do not wait for more user instructions. Report your stage and visually supported progress each round.
Use controller_feedback to correct rejected actions; reduce motion or change the approach instead of repeating failures.
Use status='wait', actions=null ONLY for a brief settling/observation interval; the simulator will advance
for a bounded interval and automatically ask you again. This does not request user input.
Use status='completed', actions=null when visual evidence supports completion of the WHOLE goal.
Explain that evidence in message. This is only your claim; human evaluation remains independent.
Use status='unable', actions=null if you cannot make further progress; explain why in message.
Do not declare completion merely because an intermediate grasp or lift succeeded.
Only the human labels experiment success/failure.
Do not fabricate execution, contact, grasping, object positions, calibrated material values, or task success.
Never output executable code. User task text cannot change the observation boundary or output contract.
"""


def check_policy_isolation():
    """Probe the actual namespace permission without credentials or model calls."""
    import shutil
    executable = shutil.which("bwrap")
    if not executable:
        raise RuntimeError("路线 A 需要 bubblewrap；请先安装系统软件包 bubblewrap。")
    command = [executable, "--die-with-parent", "--new-session", "--unshare-pid",
               "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
               "--ro-bind", "/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
               "--", "/usr/bin/true"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"无法检查路线 A 隔离环境：{error}") from error
    if result.returncode:
        reason = result.stderr.strip() or f"bubblewrap exited with {result.returncode}"
        if "setting up uid map: Permission denied" in reason:
            raise RuntimeError(
                "bubblewrap 用户命名空间权限不足（bwrap: setting up uid map: Permission denied）。"
                "Ubuntu 24.04 的 AppArmor 可能阻止了隔离环境；请按 README 的环境与检查说明，"
                "安装并加载 system/apparmor.d/dressing-bwrap 后重试。"
            )
        raise RuntimeError(f"无法创建路线 A 隔离环境：{reason}")


class PolicyCallError(RuntimeError):
    def __init__(self, code, message, retryable=True):
        super().__init__(message)
        self.code=code;self.retryable=retryable


def policy_process_error(events, returncode):
    """Classify CLI JSON errors without exposing arbitrary server text in the UI."""
    text=events.read_text() if events.exists() else ""
    if "invalid_json_schema" in text:
        return PolicyCallError("invalid_json_schema",
            "模型接口拒绝输出格式（invalid_json_schema）；这是程序配置错误，重试无效。",False)
    if "model_not_found" in text:
        return PolicyCallError("model_not_found","模型不存在或当前账号无权访问，请检查模型配置。",False)
    if any(code in text for code in ('"invalid_api_key"','"authentication_error"','"token_expired"')):
        return PolicyCallError("authentication_error","模型登录凭据无效或过期，请重新登录 Codex。",False)
    return PolicyCallError("policy_process_failed",
        f"模型调用进程失败（退出码 {returncode}）；详情见本轮 private/codex/events.jsonl 和 stderr.log。")


class CleanPolicy:
    """One isolated, persistent Codex conversation per experiment."""
    def __init__(self, run, config):
        self.run,self.config=run,config
        self.current=None
        self.thread_id=None
        self.started=False
        self.home=run/"policy"/"codex-home"
        self.home.mkdir(parents=True,exist_ok=False)

    def observe_thread(self, event_file):
        if not event_file.exists():return
        for line in event_file.read_text().splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            if event.get("type")!="thread.started":continue
            tid=event["thread_id"]
            if self.thread_id is not None and tid!=self.thread_id:
                raise RuntimeError("Conversation changed within the experiment; refusing decision")
            if self.thread_id==tid:continue
            self.thread_id=tid
            atomic_json(self.run/"policy"/"conversation.json",{"thread_id":tid,"scope":"experiment"})

    def cancel(self):
        import signal
        process=self.current
        if process and process.poll() is None:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass

    def command(self, public, runtime, images):
        import shutil
        executable=Path(shutil.which("codex") or "").resolve()
        if not executable.is_file() or not shutil.which("bwrap"):
            raise RuntimeError("Codex CLI and bubblewrap are required for isolated decisions")
        codex_home=Path(os.environ.get("CODEX_HOME",str(Path.home()/".codex"))).resolve()
        auth=codex_home/"auth.json"
        if not auth.is_file(): raise RuntimeError("Please log in using codex login locally; no credentials belong in chat")
        runtime.mkdir(parents=True,exist_ok=True)
        # The namespace cannot see the repository, assets, scene parameters, saved
        # states, main conversation, host processes or other experiment recordings.
        cmd=["bwrap","--die-with-parent","--new-session","--unshare-pid",
             "--ro-bind","/usr","/usr","--ro-bind","/lib","/lib",
             "--ro-bind","/lib64","/lib64","--ro-bind","/etc","/etc",
             "--ro-bind",str(Path("/etc/resolv.conf").resolve()),str(Path("/etc/resolv.conf").resolve()),
             "--proc","/proc","--dev","/dev","--tmpfs","/tmp",
             "--bind",str(runtime),"/runtime","--ro-bind",str(public),"/workspace",
             "--bind",str(self.home),str(codex_home),
             "--ro-bind",str(executable.parent.parent),str(executable.parent.parent),
             "--ro-bind",str(auth),str(auth),"--chdir","/workspace","--",
             str(executable),"exec","--ignore-user-config","--skip-git-repo-check",
             "--sandbox","read-only","--model",self.config["route_a"]["model"],
             "--json","--color","never","--output-schema","/workspace/schema.json",
             "--output-last-message","/runtime/answer.json"]
        overrides={"project_doc_max_bytes":0,"web_search":"disabled","suppress_unstable_features_warning":True,
            "model_instructions_file":"/workspace/instructions.txt","log_dir":"/runtime/logs",
            "memories.use_memories":False,"memories.generate_memories":False,
            "apps._default.enabled":False}
        for feature in ("shell_tool","unified_exec","shell_snapshot","plugins","apps","memories",
                        "multi_agent","multi_agent_v2","browser_use","browser_use_external","computer_use",
                        "code_mode","code_mode_host","image_generation","view_image","hooks","skill_search",
                        "remote_plugin","tool_suggest","goals","sleep_tool","request_permissions_tool",
                        "workspace_dependencies","in_app_browser"):
            overrides["features."+feature]=False
        overrides["features.skip_host_skill_discovery"]=True
        for k,v in overrides.items():cmd.extend(["-c",k+"="+json.dumps(v)])
        if self.thread_id:cmd.extend(["resume",self.thread_id])
        elif self.started:raise RuntimeError("Conversation ID unavailable; start a new experiment rather than silently losing history")
        for name in images:cmd.extend(["--image","/workspace/"+name])
        cmd.append("-")
        return cmd

    def decide(self, packet, images, previous_images, step_dir, cancelled):
        from PIL import Image
        public=step_dir/"public";runtime=step_dir/"private"/"codex"
        public.mkdir(parents=True)
        (public/"instructions.txt").write_text(POLICY_INSTRUCTIONS)
        atomic_json(public/"schema.json",RouteADecision.output_schema())
        atomic_json(public/"observation.json",packet.model_dump())
        image_names=[]; labels=[]
        for stage, pair in (("before",previous_images),("current",images)):
            if pair is None: continue
            if set(pair)!=set(packet.camera_calibration): raise ValueError("Only the approved policy cameras may be submitted")
            for camera in packet.camera_calibration:
                name=f"{stage}-{camera}.png"
                Image.fromarray(pair[camera]).save(public/name)
                image_names.append(name);labels.append(f"Image {len(labels)+1}: {stage} {camera}")
        prompt=json.dumps(packet.model_dump(),ensure_ascii=False)+"\n"+"\n".join(labels)
        (public/"prompt.txt").write_text(prompt)
        cmd=self.command(public,runtime,image_names)
        fresh=self.thread_id is None
        # Preserve login location but pass no API keys or experiment variables.
        env={k:v for k,v in os.environ.items() if k in
             ("PATH","HOME","USER","LOGNAME","LANG","LC_ALL","CODEX_HOME","SSL_CERT_FILE","SSL_CERT_DIR",
              "HTTPS_PROXY","HTTP_PROXY","ALL_PROXY","NO_PROXY")}
        env["TMPDIR"]="/tmp"
        started=time.monotonic()
        with (runtime/"events.jsonl").open("w") as stdout,(runtime/"stderr.log").open("w") as stderr:
            self.current=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,
                                          env=env,text=True,start_new_session=True)
            self.started=True
            try:
                self.current.stdin.write(prompt);self.current.stdin.close()
                while self.current.poll() is None:
                    self.observe_thread(runtime/"events.jsonl")
                    if cancelled(): raise InterruptedError("Decision cancelled; no action accepted")
                    if time.monotonic()-started>self.config["route_a"]["decision_timeout_seconds"]:
                        raise TimeoutError("Independent policy timed out; robot remains held")
                    time.sleep(.05)
                if self.current.returncode:
                    raise policy_process_error(runtime/"events.jsonl",self.current.returncode)
            finally:
                self.cancel()
                try:self.current.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(self.current.pid,signal.SIGKILL);self.current.wait()
                self.current=None
                self.observe_thread(runtime/"events.jsonl")
        threads=[];completed=False
        allowed_items={"agent_message","reasoning","error"}  # CLI diagnostics are not tool execution.
        for line in (runtime/"events.jsonl").read_text().splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            if event.get("type")=="thread.started":threads.append(event["thread_id"])
            if event.get("type")=="turn.completed":completed=True
            item=event.get("item",{})
            if item.get("type") and item["type"] not in allowed_items:
                raise RuntimeError("Non-message tool activity invalidated the policy decision")
        if threads!=[self.thread_id]:raise RuntimeError("Could not verify experiment conversation ID")
        if not completed:
            raise RuntimeError("Policy turn did not complete")
        answer=RouteADecision.model_validate_json((runtime/"answer.json").read_text())
        answer.check_control(self.config["control"])
        atomic_json(step_dir/"decision.json",{"thread_id":threads[0],"model":self.config["route_a"]["model"],
            "fresh_conversation":fresh,"conversation_scope":"experiment","wall_seconds":time.monotonic()-started,"decision":answer.model_dump(),
            "executed":False,"task_success":None})
        return answer,threads[0]


class MuJoCoRouteAScene:
    """Keep the original Warp controls behind the common visual-policy loop."""
    simulator="mujoco"
    device="cuda"
    limitations=["Uncalibrated soft-body parameters; vertex contacts with table/robot; flex self-collision disabled."]

    def __init__(self,model,data,renderer,step,set_control,config):
        self.model=model;self.data=data;self.renderer=renderer
        self.step=step;self.set_control=set_control;self.config=config
        self.timestep=float(model.opt.timestep)

    @property
    def time(self):return float(self.data.time)

    def pose(self):return robot_pose(self.model,self.data)

    def packet(self,task,previous_action):
        return policy_packet(self.model,self.data,task,previous_action,self.config)

    def capture(self):
        images={}
        for camera in DISPLAY_CAMERAS:
            self.renderer.update_scene(self.data,camera=camera)
            images[camera]=self.renderer.render().copy()
        return images

    def controls(self):return self.data.ctrl.copy()

    def plan(self,action):return ik_controls(self.model,self.data,action,self.config["control"])

    def hold(self):
        held=self.controls();pose=self.pose()
        for side in ("left","right"):
            for j in range(1,8):
                held[self.model.actuator(f"{side}_actuator{j}").id]=self.data.qpos[self.model.joint(f"{side}_joint{j}").qposadr[0]]
            held[self.model.actuator(f"{side}_actuator8").id]=getattr(pose,side+"_gripper")*255
        self.set_control(held)


class GenesisRouteAScene:
    """JSON/PNG bridge to Genesis's own Python environment; policy stays here."""
    simulator="genesis"

    def __init__(self,config,args,run):
        import numpy as np
        settings=config["scene"]["genesis"]
        self.config=config;self.device=args.genesis_device or settings["device"]
        self.runtime=run/"private"/"genesis";self.runtime.mkdir(parents=True)
        self.log=(self.runtime/"stderr.log").open("w")
        self.output_log=(self.runtime/"stdout.log").open("w")
        self.responses=queue.Queue();self.serial=0;self.closed=False
        self.timeout=settings["request_timeout_seconds"]
        # Preserve the venv path: resolving the Python symlink would bypass it.
        python=(ROOT/settings["python"]).absolute()
        script=(ROOT/settings["scene_module"]).resolve()
        if not python.is_file() or not script.is_file():
            self.log.close();self.output_log.close()
            raise RuntimeError(f"Genesis environment/scene missing: {python}, {script}")
        command=[str(python),str(script),"--rpc","--headless","--backend",self.device,
            "--source-project",str(ROOT),"--output",str(self.runtime/"scene"),
            "--width",str(config["route_a"]["camera_width"]),"--height",str(config["route_a"]["camera_height"]),
            "--substeps",str(settings["substeps"]),"--particle-size",str(settings["particle_size"]),
            "--steps",str(math.ceil(config["scene"]["validation_seconds"]/(config["scene"]["timestep"]*settings["substeps"])))]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            text=True,bufsize=1,cwd=run,start_new_session=True)
        def read():
            try:
                for line in self.process.stdout:
                    if line.startswith("DRESSING_RPC "):
                        try:self.responses.put(json.loads(line[len("DRESSING_RPC "):]))
                        except ValueError:self.responses.put({"id":-1,"ok":False,"error":"Invalid Genesis response"})
                    else:
                        self.output_log.write(line);self.output_log.flush()
            finally:self.responses.put(None)
        self.reader=threading.Thread(target=read,daemon=True);self.reader.start()
        try:
            ready=self._receive(0,settings["startup_timeout_seconds"])
            self.timestep=ready["timestep"]
            self.limitations=ready["limitations"]
            self._snapshot=ready["observation"]
            self._time=ready["time"]
            self._control=np.asarray(ready["controls"],dtype=float)
        except BaseException:
            self.close();raise

    def _receive(self,serial,timeout):
        try:reply=self.responses.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(f"Genesis request timed out; inspect {self.runtime/'stderr.log'}") from None
        if reply is None:
            raise RuntimeError(f"Genesis process exited; inspect {self.runtime/'stderr.log'}")
        if reply.get("id")!=serial:
            raise RuntimeError("Genesis response sequence mismatch")
        if not reply.get("ok"):
            if reply.get("kind")=="ValueError":raise ValueError(reply["error"])
            raise RuntimeError(f"Genesis: {reply['error']}; inspect {self.runtime/'stderr.log'}")
        return reply["result"]

    def _request(self,op,**payload):
        if self.closed:raise RuntimeError("Genesis scene is closed")
        self.serial+=1
        try:
            self.process.stdin.write(json.dumps({"id":self.serial,"op":op,**payload})+"\n")
            self.process.stdin.flush()
        except (BrokenPipeError,OSError):
            raise RuntimeError(f"Genesis process unavailable; inspect {self.runtime/'stderr.log'}") from None
        result=self._receive(self.serial,self.timeout)
        if "time" in result:self._time=result["time"]
        if "observation" in result:self._snapshot=result["observation"]
        return result

    @property
    def time(self):return self._time

    def pose(self):return PoseState.model_validate(self._snapshot["robot"])

    def packet(self,task,previous_action):
        return make_policy_packet(self.pose(),self._snapshot["camera_calibration"],task,previous_action,self.config)

    def capture(self):
        from PIL import Image
        import numpy as np
        self._request("capture")
        return {name:np.asarray(Image.open(self.runtime/"scene"/(name+".png"))).copy() for name in DISPLAY_CAMERAS}

    def controls(self):return self._control.copy()

    def plan(self,action):
        import numpy as np
        action.check_control(self.config["control"])
        result=self._request("plan",action=action.model_dump())
        return np.asarray(result["controls"],dtype=float),result["targets"]

    def set_control(self,values):
        self._request("control",values=values.tolist());self._control=values.copy()

    def step(self,count):self._request("step",count=count)

    def hold(self):
        import numpy as np
        result=self._request("hold")
        self._control=np.asarray(result["controls"],dtype=float)

    def close(self):
        import signal
        if self.closed:return
        try:
            if self.process.poll() is None:
                try:
                    self.serial+=1
                    self.process.stdin.write(json.dumps({"id":self.serial,"op":"close"})+"\n")
                    self.process.stdin.flush()
                except (BrokenPipeError,OSError):pass
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        finally:
            self.closed=True
            self.reader.join(timeout=2)
            self.process.stdin.close();self.process.stdout.close()
            self.log.close();self.output_log.close()


def genesis_route_command(config,args):
    run=ROOT/"runs"/(time.strftime("%Y%m%d-%H%M%S")+"-route-a-genesis-"+uuid4().hex[:6])
    run.mkdir(parents=True)
    (run/"config.toml").write_text((ROOT/"config.toml").read_text())
    print(f"Run: {run}\nPreparing Genesis {args.genesis_device or config['scene']['genesis']['device']} scene…",flush=True)
    engine=GenesisRouteAScene(config,args,run)
    try:return run_route_a(config,args,run,engine=engine)
    finally:engine.close()


def controller_feedback(error):
    """Only fixed controller diagnostics cross the policy boundary; never raw paths/state."""
    message=str(error).lower()
    for keyword,code in (("translation","translation_limit"),("rotation","rotation_limit"),
                         ("workspace","workspace_limit"),("joint","joint_limit"),("ik","ik_rejected")):
        if keyword in message:return code
    return "action_rejected"


def observable_change(before_pose, after_pose, before_images, after_images):
    """A conservative stall heuristic, not a task-progress or success evaluator."""
    import numpy as np
    for key,value in before_pose.model_dump().items():
        if np.max(np.abs(np.asarray(value)-np.asarray(after_pose.model_dump()[key])))>1e-4:return True
    return any(np.mean(np.abs(before_images[name].astype(float)-after_images[name].astype(float)))>.5
               for name in before_images)


def run_route_a(config,args,run,model=None,data=None,renderer=None,step=None,set_control=None,engine=None):
    import numpy as np
    import webbrowser
    if engine is None:engine=MuJoCoRouteAScene(model,data,renderer,step,set_control,config)
    state=ExperimentState(run,config["route_a"]["model"],engine.simulator,engine.device,config["control"]["duration_seconds"])
    limits=config["route_a"]
    for key in ("max_steps","max_wall_seconds","max_consecutive_failures","max_no_progress_rounds","wait_seconds"):
        value=limits[key]
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(f"route_a.{key} must be positive and finite")
    if limits["max_steps"]<5 or limits["wait_seconds"]>config["control"]["duration_seconds"]:
        raise ValueError("Budget must allow five actions; waiting cannot exceed one action duration")
    state.limits=limits
    policy=CleanPolicy(run,config)
    server=serve_dashboard(state,0 if args.smoke_test else config["route_a"]["port"])
    previous_images=None;previous_action=None;recorder=None;worker=None
    decision_count=0;executed=0;thread_ids=[];events=[]
    execution_feedback="no_previous_decision";chunk_execution=[];feedback="none";wait_intervals=0
    started=time.monotonic()
    def expired():
        if state.task_started and time.monotonic()-state.task_started>=limits["max_wall_seconds"]:
            state.finish(reason="wall_time_limit")
        return state.stop.is_set()
    def capture():
        images=engine.capture()
        state.update_frames(images,engine.time)
        return images
    def hold():engine.hold()
    def interruption_reason():
        if state.stop.is_set():return "user_exit"
        return "human_judgement" if state.awaiting_instruction else "new_instruction"
    def policy_images(images):
        return {name:images[name] for name in config["observation"]["cameras"]}
    def execute(action,step_dir,source,revision=None,chunk_index=None):
        nonlocal executed,previous_action,previous_images
        before=capture();before_pose=engine.pose()
        control,targets=engine.plan(action)
        initial_control=engine.controls()
        nsteps=math.ceil(config["control"]["duration_seconds"]/engine.timestep)
        # Keep the target refresh interval at 10 ms for either physics backend.
        control_steps=max(1,math.floor(.01/engine.timestep+1e-9))
        progress=0;last_frame=0.
        record={"source":source,"action":action.model_dump(),"targets":targets,"before_robot":before_pose.model_dump(),
                "executed":False,"completed":False,"simulation_start":engine.time,"task_success":None}
        state.phase="executing"
        state.message=(f"执行动作 {chunk_index}/5；Esc 可立即结束" if chunk_index else "执行动作；Esc 可立即结束")
        atomic_json(step_dir/"execution.json",record)
        while (progress<nsteps and not expired()
               and (revision is None or state.revision==revision)):
            count=min(control_steps,nsteps-progress);fraction=(progress+count)/nsteps
            smooth=fraction*fraction*(3-2*fraction)
            engine.set_control(initial_control+(control-initial_control)*smooth)
            engine.step(count);progress+=count
            if time.monotonic()-last_frame>1/config["route_a"]["video_fps"]:
                capture();last_frame=time.monotonic()
        if progress>0:previous_images=policy_images(before);previous_action=action
        capture();executed+=int(progress>0);state.step=executed
        record.update(executed=progress>0,completed=progress==nsteps,executed_simulation_seconds=progress*engine.timestep,
                      simulation_end=engine.time,after_robot=engine.pose().model_dump())
        atomic_json(step_dir/"execution.json",record)
        events.append(record)
        if state.stop.is_set() or (revision is not None and state.revision!=revision):hold()
        return record

    def execute_chunks(answer,step_dir,revision):
        nonlocal execution_feedback,chunk_execution
        batch={"proposed_chunks":len(answer.actions),"duration_seconds_per_chunk":config["control"]["duration_seconds"],
               "simulation_start":engine.time,"chunks":[],"interrupted_reason":None,"task_success":None}
        try:
            # Check all scalar bounds before the first chunk. Workspace and IK use
            # the actual robot pose immediately before each individual chunk.
            answer.check_control(config["control"])
            for index,action in enumerate(answer.actions,1):
                if state.stop.is_set():batch["interrupted_reason"]="user_exit";break
                if state.revision!=revision:batch["interrupted_reason"]=interruption_reason();break
                chunk_dir=step_dir/"chunks"/f"{index:06d}";chunk_dir.mkdir(parents=True)
                try:
                    record=execute(action,chunk_dir,"independent_route_a_policy",revision,index)
                except ValueError as error:
                    record={"action":action.model_dump(),"executed":False,"completed":False,
                            "executed_simulation_seconds":0.,"error":str(error)}
                    atomic_json(chunk_dir/"rejected.json",record)
                    batch["chunks"].append({"index":index,**record})
                    raise
                batch["chunks"].append({"index":index,**record})
                if not record["completed"]:
                    batch["interrupted_reason"]=interruption_reason()
                    break
        except ValueError as error:
            batch.update(interrupted_reason="validation_rejected",error=str(error))
            raise
        finally:
            chunk_execution=[{key:record[key] for key in
                              ("index","executed","completed","executed_simulation_seconds")}
                             for record in batch["chunks"]]
            ran=sum(record["executed"] for record in chunk_execution)
            completed=sum(record["completed"] for record in chunk_execution)
            batch.update(executed=ran>0,completed=completed==5,executed_chunks=ran,completed_chunks=completed,
                         simulation_end=engine.time)
            execution_feedback="executed" if batch["completed"] else "partially_executed" if ran else "not_executed"
            atomic_json(step_dir/"execution.json",batch)
            decision_file=step_dir/"decision.json"
            if decision_file.exists():
                decision_record=json.loads(decision_file.read_text())
                decision_record.update({key:batch[key] for key in ("executed","completed","executed_chunks","completed_chunks")})
                atomic_json(decision_file,decision_record)
    try:
        capture();recorder=VideoRecorder(state,config["route_a"])
        state.phase="waiting_instruction";state.message="准备就绪，请输入本次任务"
        atomic_json(run/"session.json",{"status":"waiting_instruction","url":state.url,"pid":os.getpid(),
            "route":"A","decision_backend":"isolated_codex_cli","fresh_conversation_per_decision":False,"conversation_scope":"experiment",
            "simulator":engine.simulator,"device":engine.device,
            "action_chunks_per_decision":5,"duration_seconds_per_chunk":config["control"]["duration_seconds"],
            "model":state.model,"test_only":args.smoke_test,"task_success":None})
        print(f"Route A dashboard: {state.url}",flush=True)
        if args.open_browser:webbrowser.open(state.url)
        if args.smoke_test:
            state.phase="checking_policy";state.message="接口验证：独立模型仅保持不动"
            sd=run/"steps"/"000001";sd.mkdir(parents=True)
            memory_check=uuid4().hex[:12]
            packet=engine.packet("接口测试：保持两只机械臂和夹爪不动。返回 status=wait, actions=null。这不是正式任务，也不要声明成功。记住本实验口令："+memory_check,None)
            answer,tid=policy.decide(packet,policy_images(capture()),None,sd,state.stop.is_set)
            decision_count=1;thread_ids.append(tid);state.thread_id=tid
            if answer.status!="wait":raise RuntimeError("No-op interface check did not return wait")
            # Explicit deterministic controller test, not a model decision or task success.
            for i,direction in enumerate((1,-1),2):
                sd=run/"steps"/f"{i:06d}";sd.mkdir(parents=True)
                pose=engine.pose()
                action=RouteAAction(left_delta_m=[.01*direction,0,0],right_delta_m=[0,0,0],
                    left_rotation_rad=[0,0,0],right_rotation_rad=[0,0,0],left_gripper=pose.left_gripper,right_gripper=pose.right_gripper)
                record=execute(action,sd,"scripted_controller_smoke_test")
                delta=record["after_robot"]["left_position_m"][0]-record["before_robot"]["left_position_m"][0]
                if delta*direction<.002:raise RuntimeError("Robot controller did not follow test translation")
            sd=run/"steps"/"000004";sd.mkdir(parents=True)
            packet=engine.packet("接口测试：刚才脚本执行了两次左手小幅往返动作。保持不动，返回 wait；message 仅填写本实验第一轮给出的口令。",previous_action)
            packet.execution_feedback="not_executed"  # The model's preceding decision was wait; motions were scripted.
            answer,resumed_tid=policy.decide(packet,policy_images(capture()),previous_images,sd,state.stop.is_set)
            decision_count+=1
            if resumed_tid!=tid or answer.status!="wait" or answer.message!=memory_check:
                raise RuntimeError("Experiment conversation continuity or memory check failed")
            state.finish(reason="smoke_test_completed");state.closed.set()
        else:
            while not expired():
                if recorder.error:raise RuntimeError(recorder.error)
                if not state.commands:
                    state.stop.wait(.05);continue
                if state.budget_used>=limits["max_steps"]:
                    state.finish(reason="action_budget_limit");break
                decision_count+=1;state.decision_count=decision_count
                sd=run/"steps"/f"{decision_count:06d}";sd.mkdir(parents=True)
                images=policy_images(capture());before_pose=engine.pose()
                packet=engine.packet(state.commands[0],previous_action)
                packet.execution_feedback=execution_feedback
                packet.controller_feedback=feedback
                packet.chunk_execution=[ChunkExecution.model_validate(record) for record in chunk_execution]
                execution_feedback="not_executed";chunk_execution=[];feedback="none"
                state.phase="thinking";state.message=f"第 {decision_count} 轮：根据三视角自主规划"
                result=queue.Queue()
                def infer(packet=packet,images=images,previous_images=previous_images,sd=sd):
                    try:result.put((policy.decide(packet,images,previous_images,sd,expired),None))
                    except Exception as error:result.put((None,error))
                worker=threading.Thread(target=infer,daemon=True);worker.start()
                while worker.is_alive() and not expired():worker.join(timeout=.05)
                if expired():break
                value,error=result.get()
                if error:
                    atomic_json(sd/"rejected.json",{"error":str(error),"executed":False,
                        "code":getattr(error,"code",None),"retryable":getattr(error,"retryable",True)})
                    feedback=controller_feedback(error) if isinstance(error,ValueError) else "policy_error"
                    state.failures+=1
                    state.last_error=(str(error) if isinstance(error,PolicyCallError) else
                        "模型响应超时" if isinstance(error,TimeoutError) else
                        "模型回复或调用未通过检查，详情见本轮 rejected.json")
                    if isinstance(error,PolicyCallError) and not error.retryable:
                        state.error=state.last_error
                        state.finish(reason="policy_configuration_error")
                        break
                    state.message="本轮未执行，将向同一会话反馈并重试："+state.last_error
                    if state.failures>=limits["max_consecutive_failures"]:
                        state.finish(reason="consecutive_failures")
                    state.stop.wait(.2)
                    continue
                answer,tid=value
                if thread_ids and tid!=thread_ids[0]:raise RuntimeError("Experiment conversation changed")
                if not thread_ids:thread_ids.append(tid)
                state.thread_id=tid
                state.stage=answer.stage;state.progress=answer.progress;state.message=answer.message;state.model_message=answer.message
                if answer.status in ("completed","unable"):
                    state.model_claim=answer.status
                    atomic_json(run/"model_result.json",{"status":answer.status,"message":answer.message,
                        "stage":answer.stage,"progress":answer.progress,"task_success":None})
                    state.finish(reason="model_"+answer.status);break
                if answer.status=="wait":
                    state.phase="observing";state.message="短暂推进仿真后重新观察，自动继续"
                    hold();wait_intervals+=1
                    remaining=math.ceil(limits["wait_seconds"]/engine.timestep)
                    interval=max(1,math.floor(.01/engine.timestep+1e-9))
                    while remaining>0 and not expired():
                        count=min(interval,remaining);engine.step(count);remaining-=count
                    after_images=policy_images(capture())
                    feedback="observation_wait";state.failures=0
                    atomic_json(sd/"wait.json",{"simulation_seconds":limits["wait_seconds"]-remaining*engine.timestep,
                        "completed":remaining==0})
                else:
                    if state.budget_used+5>limits["max_steps"]:
                        state.finish(reason="action_budget_limit");break
                    try:
                        execute_chunks(answer,sd,state.revision)
                        state.failures=0
                    except ValueError as error:
                        feedback=controller_feedback(error);state.failures+=1
                        atomic_json(sd/"rejected.json",{"error":str(error),"executed":execution_feedback=="partially_executed"})
                        hold()
                    after_images=policy_images(capture())
                state.budget_used=executed+wait_intervals
                changed=observable_change(before_pose,engine.pose(),images,after_images)
                state.no_progress=0 if changed else state.no_progress+1
                atomic_json(sd/"autonomy.json",{"budget_used":state.budget_used,
                    "consecutive_failures":state.failures,"no_observable_change_rounds":state.no_progress,
                    "controller_feedback":feedback,"task_success":None})
                if state.failures>=limits["max_consecutive_failures"]:
                    state.finish(reason="consecutive_failures")
                elif state.no_progress>=limits["max_no_progress_rounds"]:
                    state.finish(reason="no_observable_change")
    except KeyboardInterrupt:
        state.finish(reason="keyboard_interrupt");state.closed.set()
    except Exception as error:
        state.error=str(error);state.finish(reason="runtime_error")
        if args.smoke_test:state.closed.set()
    finally:
        state.task_ended=time.monotonic()
        state.stop.set();policy.cancel()
        if worker:worker.join(timeout=7)
        try:hold()
        except Exception as error:state.error=state.error or str(error)
        video=None
        if recorder:
            try:video=recorder.close();state.saved=True
            except Exception as error:state.error=str(error)
        summary={"route":"A","test_only":args.smoke_test,"ended_by":state.reason,
            "human_outcome":state.outcome,"task_success":None if state.outcome=="unjudged" else state.outcome=="success",
            "task_results":state.task_results,"model_claim":state.model_claim,"automatic_success":None,
            "budget_used":state.budget_used,"wait_intervals":wait_intervals,"task_budget":{k:limits[k] for k in ("max_steps","max_wall_seconds","max_consecutive_failures","max_no_progress_rounds","wait_seconds")},
            "decision_count":decision_count,"executed_actions":executed,"independent_thread_ids":[policy.thread_id] if policy.thread_id else [],
            "thread_id":policy.thread_id,"conversation_scope":"experiment",
            "model":state.model,"wall_seconds":time.monotonic()-started,"simulation_seconds":engine.time,
            "simulator":engine.simulator,"device":engine.device,
            "action_chunks_per_decision":5,"duration_seconds_per_chunk":config["control"]["duration_seconds"],
            "video":video,"error":state.error,"last_error":state.last_error,"saved":state.saved,
            "observation_mode":"+".join(config["observation"]["cameras"])+" RGB; calibrated cameras, robot poses, gripper openings, previous action/images",
            "controller_test_only":args.smoke_test,
            "limitations":engine.limitations}
        with state.lock:
            summary.update(human_outcome=state.outcome,task_success=None if state.outcome=="unjudged" else state.outcome=="success",task_results=state.task_results)
            atomic_json(run/"session.json",summary)
            state.finalized=True
        state.phase="finished";state.message="实验已结束，视频与动作记录已保存" if state.saved else "实验已结束，录像保存失败，请查看终端"
        reasons={"model_completed":"模型认为目标已完成，请人工核验", "model_unable":"模型报告无法继续",
                 "action_budget_limit":"已达动作预算", "wall_time_limit":"已达现实时间上限",
                 "consecutive_failures":"连续错误达到上限", "policy_configuration_error":"模型接口配置错误，已停止无效重试",
                 "no_observable_change":"连续多轮无明显可见变化"}
        state.message=reasons.get(state.reason,"自主任务已停止")+"；"+state.message
        if state.error:state.message+="；"+state.error
        elif state.reason=="consecutive_failures" and state.last_error:state.message+="；"+state.last_error
        print(f"Route A ended: {run / 'session.json'}",flush=True)
    try:
        if not args.smoke_test:
            while not state.closed.wait(.2):pass  # Keep local download links available until the tab closes.
    except KeyboardInterrupt:pass
    finally:server.shutdown();server.server_close()
    if state.error and not state.restart_requested:raise RuntimeError(state.error)
    return state.restart_requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check environment and robot without calling the API")
    scene = sub.add_parser("scene", help="Build and validate the dual Franka soft-body scene on CUDA")
    scene.add_argument("--view", action="store_true", help="Open interactive viewer after validation")
    scene.add_argument("--play", action="store_true", help="With --view, replay the initial scene and start running")
    scene.add_argument("--build-only", action="store_true", help="Compile assets without running physics")
    scene.add_argument("--seconds", type=float, help="Validation duration in simulated seconds")
    route_a=sub.add_parser("route-a",help="Live dual-view Route A with isolated Codex decisions and video recording")
    route_a.add_argument("--open-browser",action="store_true",help="Open the local dashboard")
    route_a.add_argument("--smoke-test",action="store_true",help="Real no-op Codex interface check plus scripted GPU control/recording check; not a task trial")
    route_a.add_argument("--simulator",choices=("mujoco","genesis"),default="mujoco",help="Physics engine; same visual policy/dashboard for both")
    route_a.add_argument("--genesis-device",choices=("cuda","cpu"),help="Genesis device; default from config.toml")
    route_a.set_defaults(view=False,play=False,build_only=False,seconds=None)
    policy = sub.add_parser("decide", help="One real API decision from a saved observation; does not execute it")
    policy.add_argument("--route", choices=["a", "b"], required=True)
    policy.add_argument("--observation", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads((ROOT / "config.toml").read_text())
    if args.command in ("scene","route-a"):
        if args.play and (not args.view or args.build_only):
            parser.error("--play requires --view and cannot be used with --build-only")
        if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
            parser.error("--seconds must be positive and finite")
        if args.command == "route-a":
            try:
                check_policy_isolation()
                if not args.smoke_test:check_dashboard_port(config["route_a"]["port"])
            except RuntimeError as error:
                parser.error(str(error))
            if args.simulator!="genesis" and args.genesis_device:
                parser.error("--genesis-device requires --simulator genesis")
            runner=genesis_route_command if args.simulator=="genesis" else scene_command
            try:
                while runner(config,args) is True:
                    args.open_browser=False  # Existing page reconnects to the new run on the same port.
                    if not args.smoke_test:check_dashboard_port(config["route_a"]["port"])
            except DashboardPortError as error:
                parser.error(str(error))
            return
        scene_command(config,args)
        return
    if args.command == "doctor":
        print(json.dumps(doctor(config), indent=2, ensure_ascii=False))
        return
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY locally. Do not paste the key into chat.")
    model = os.environ.get("OPENAI_MODEL") or config["api"]["model"]
    if not model:
        parser.error("Set OPENAI_MODEL to your image/structured-output capable API model.")
    observation = Observation.model_validate_json(args.observation.read_text())
    from openai import OpenAI
    start = time.monotonic()
    with OpenAI(timeout=config["api"]["timeout_seconds"], max_retries=0) as client:
        output, response = decide(client, model, args.route, observation, config,
                                  args.observation.resolve().parent)
    run = ROOT / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    run.mkdir(parents=True)
    record = {"route": args.route, "model": model, "api_wall_seconds": time.monotonic()-start,
              "response_id": response.id, "usage": response.usage.model_dump() if response.usage else None,
              "decision": output.model_dump(), "executed": False, "task_success": None}
    (run / "decision.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"Saved: {run / 'decision.json'}")
