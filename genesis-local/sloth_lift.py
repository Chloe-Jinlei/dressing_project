"""Scripted dual-arm manipulation using contact-gated, local PBD attachments.

Attachments model a no-slip grasp; they do not validate Coulomb friction or
gripper holding force. Every object motion is produced by stepping Genesis.
"""
from __future__ import annotations

import json
import numpy as np
import quadrants as qd
import genesis as gs
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


def array(tensor):
    return tensor.detach().cpu().numpy().copy()


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()


class LiftExperiment:
    def __init__(self, scene, robots, objects, cameras, config, report, run, args):
        from genesis.utils.video_encoder import VideoEncoder

        self.scene, self.robots, self.objects, self.cameras = scene, robots, objects, cameras
        self.report, self.run, self.args = report, run, args
        self.table = config["scene"]["table_height"]
        self.dt = report["genesis"]["step_dt_s"]
        self.step_count = 0
        self.phase = "settle"
        self.hands = {side: robot.get_link("hand") for side, robot in robots.items()}
        self.hand_quat = {side: [0, 1, 0, 0] for side in robots}
        self.fingers = {side: [robot.get_link(name) for name in ("left_finger", "right_finger")]
                        for side, robot in robots.items()}
        self.attached = {}
        self.used_particles = set()
        self.targets = {}
        self.samples = []
        self.next_frame_time = 0.
        self.max_arm_contacts = 0
        self.arm_contacts = 0
        self.font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20)
        self.small_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16)
        filename = "dressing-process.mp4" if args.task == "dress-sloth" else "lift-process.mp4"
        self.video = VideoEncoder(str(run / filename), args.record_fps, codec="libx264", bitrate=3.)
        self.log = (run / "trajectory.jsonl").open("w")
        self.task = report["lift_task"] = {
            "controller": "scripted IK and joint position control; simulation geometry used to select limb grasps",
            "grasp_model": "contact-gated local PBD attachments to the actual finger links",
            "not_a_friction_grasp_validation": True,
            "sloth_material": report["sloth_soft_body"],
            "requested_lift_m": args.lift_height,
            "hold_seconds": args.hold_seconds,
            "contact_tolerance_m": .004,
            "object_position_setters_used": False,
            "robot_position_setters_during_task": False,
            "robot_gravity_compensation": 1.0,
            "phases": [], "attachments": self.attached,
        }
        report["limitations"][0] = "Scripted dual-arm lift only; no GPT policy, garment manipulation, or dressing success evaluation."
        report["limitations"].extend([
            "Lift uses local no-slip PBD attachments activated only after both fingertip pads approach surface particles within 4 mm and the gripper closes. This is not validation of frictional grasping or holding force.",
            "The lift controller is scripted and uses simulator geometry; it is not a vision-only or GPT policy.",
            "PBD link attachments do not apply a reciprocal grasp-constraint load to the robot; actuator load capacity is not validated.",
        ])
        with np.load(run / "assets" / "physics_meshes.npz") as physics:
            self.rest = physics["sloth_points"].copy()
            tetra = physics["sloth_tetra"]
        self.tetra = tetra.copy()
        self.rest_tetra_volumes = self.tetra_volumes(self.rest)
        from dressing_scene import tetra_boundary
        self.surface_indices, _ = tetra_boundary(self.rest, tetra)
        self.edges = np.unique(np.sort(np.concatenate([
            tetra[:, [a, b]] for a in range(4) for b in range(a + 1, 4)]), axis=1), axis=0)
        self.rest_lengths = np.linalg.norm(self.rest[self.edges[:, 0]] - self.rest[self.edges[:, 1]], axis=1)
        self.grip = {side: np.array([.04, .04]) for side in robots}
        for robot in robots.values():
            robot.set_dofs_kp([500, 500], dofs_idx_local=[7, 8])
            robot.set_dofs_kv([20, 20], dofs_idx_local=[7, 8])
        print(f"Lift sloth material: PBD.Elastic {report['sloth_soft_body']['parameters']}", flush=True)

    def persist(self):
        self.task["completed_steps"] = self.step_count
        (self.run / "report.json").write_text(json.dumps(self.report, indent=2))

    def start_phase(self, name):
        self.phase = name
        self.task["phases"].append({"name": name, "start_seconds": self.step_count * self.dt})
        self.persist()
        print(f"Lift phase: {name} (t={self.step_count * self.dt:.2f}s)", flush=True)

    def tcp(self, side):
        link = self.hands[side]
        return array(link.get_pos(relative=False)) + rotation(array(link.get_quat(relative=False))) @ np.array([0, 0, .1029])

    def sample(self):
        points = array(self.objects["sloth"].get_particles_pos())
        velocities = array(self.objects["sloth"].get_particles_vel())
        if not np.isfinite(points).all() or not np.isfinite(velocities).all():
            raise RuntimeError("Sloth has non-finite particle state")
        stretch = np.linalg.norm(points[self.edges[:, 0]] - points[self.edges[:, 1]], axis=1) / self.rest_lengths
        volumes = self.tetra_volumes(points)
        sample = {
            "step": self.step_count, "seconds": self.step_count * self.dt, "phase": self.phase,
            "sloth_center_m": points.mean(0).tolist(),
            "sloth_min_z_m": float(points[:, 2].min()),
            "table_clearance_m": float(points[:, 2].min() - self.table),
            "sloth_max_speed_m_s": float(np.linalg.norm(velocities, axis=1).max()),
            "max_edge_stretch_ratio": float(stretch.max()),
            "sloth_volume_ratio": float(np.abs(volumes).sum() / np.abs(self.rest_tetra_volumes).sum()),
            "inverted_tetra_rest_volume_fraction": float(np.abs(self.rest_tetra_volumes[volumes * self.rest_tetra_volumes < 0]).sum()
                                                        / np.abs(self.rest_tetra_volumes).sum()),
            "gripper_tcp_m": {side: self.tcp(side).tolist() for side in self.robots},
            "robot_qpos": {side: array(robot.get_qpos()).tolist() for side, robot in self.robots.items()},
        }
        if not all(np.isfinite(sample["robot_qpos"][side]).all() for side in self.robots):
            raise RuntimeError("Robot has non-finite state")
        sample["inter_arm_contact_count"] = self.arm_contacts
        self.samples.append(sample)
        self.log.write(json.dumps(sample) + "\n")
        self.log.flush()
        return sample

    def tetra_volumes(self, points):
        p = points[self.tetra]
        return np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6

    def frame(self):
        images = []
        for name in ("overview", "lift"):
            rgb, *_ = self.cameras[name].render()
            images.append(Image.fromarray(np.asarray(rgb)))
        w, h = images[0].size
        canvas = Image.new("RGB", (2 * w, h + 96), "#111827")
        canvas.paste(images[0], (0, 48))
        canvas.paste(images[1], (w, 48))
        draw = ImageDraw.Draw(canvas)
        names = {"settle": "初始静置", "left_approach": "左臂接近", "right_approach": "右臂接近",
                 "descend": "双臂下降到抓取点", "align_grasp": "根据表面重新对齐", "reopen_gripper": "重新张爪",
                 "close_grippers": "闭合夹爪 / 检查接触",
                 "grasp_hold": "局部抓取约束激活", "lift": "双臂同步抬升", "airborne_hold": "空中保持"}
        names.update({"transfer_sloth": "移动到穿衣位置", "support_sloth": "桌面支撑树懒",
                      "regrasp_rear_feet": "重新抓持后脚", "raise_rear_feet": "提起后脚",
                      "orient_sloth": "调整树懒头部朝向",
                      "grasp_hem": "抓取衣服前片下摆", "open_hem": "撑开衣服下摆",
                      "grasp_collar": "抓取领口", "thread_collar": "将领口拉过头部",
                      "grasp_right_sleeve": "抓取右袖口", "thread_right_sleeve": "整理右袖口",
                      "grasp_left_sleeve": "抓取左袖口", "thread_left_sleeve": "整理左袖口",
                      "align_to_hem": "树懒抬升到开口高度",
                      "insert_sloth": "从下摆穿入树懒", "settle_garment": "整理软体衣服",
                      "release_sloth": "释放树懒", "dressed_hold": "检查领口和袖口"})
        task_name = "双臂给软体树懒穿衣" if self.args.task == "dress-sloth" else "双臂抓取软体树懒"
        title = f"{task_name}  |  {names.get(self.phase, self.phase)}  |  t={self.step_count * self.dt:.2f}s"
        draw.text((16, 8), title, fill="white", font=self.font)
        clearance = array(self.objects["sloth"].get_particles_pos())[:, 2].min() - self.table
        draw.text((16, h + 55), f"最低点离桌：{clearance * 100:.1f} cm    抓取模型：接触触发的局部约束    仿真原速", fill="#e5e7eb", font=self.small_font)
        self.video.write(np.asarray(canvas))
        return canvas

    def step(self):
        if not self.args.headless and not self.scene.viewer.is_alive():
            raise RuntimeError("Viewer closed before the lift task completed")
        self.before_physics_step()
        self.scene.step()
        self.after_physics_step()
        self.step_count += 1
        contacts = self.robots["left"].get_contacts(with_entity=self.robots["right"])
        self.arm_contacts = len(contacts["geom_a"])
        self.max_arm_contacts = max(self.max_arm_contacts, self.arm_contacts)
        if self.arm_contacts:
            self.report["inter_arm_collision"] = {
                "phase": self.phase, "seconds": self.step_count * self.dt,
                "positions_m": array(contacts["position"]).tolist(),
                "links_a": [self.scene.rigid_solver.links[int(idx)].name for idx in array(contacts["link_a"])],
                "links_b": [self.scene.rigid_solver.links[int(idx)].name for idx in array(contacts["link_b"])],
            }
            self.persist()
            raise RuntimeError("Dual-arm path caused an inter-arm collision")
        if self.step_count % 20 == 0:
            self.sample()
        if self.step_count * self.dt + 1e-9 >= self.next_frame_time:
            self.frame()
            self.next_frame_time += 1 / self.args.record_fps
        if self.step_count % 200 == 0:
            print(f"Lift step {self.step_count}, phase={self.phase}", flush=True)

    def after_physics_step(self):
        pass

    def before_physics_step(self):
        pass

    def wait(self, seconds):
        for _ in range(round(seconds / self.dt)):
            self.step()

    def save_stage(self, name):
        self.frame().save(self.run / f"stage-{name}.png")
        for camera_name in ("overview", "lift"):
            rgb, *_ = self.cameras[camera_name].render()
            Image.fromarray(np.asarray(rgb)).save(self.run / f"{name}-{camera_name}.png")
        self.persist()

    def solve(self, side, tcp):
        robot = self.robots[side]
        qpos, error = robot.inverse_kinematics(
            link=self.hands[side], pos=tcp, quat=self.hand_quat[side], local_point=[0, 0, .1029],
            init_qpos=robot.get_qpos(), max_samples=20, max_solver_iters=50,
            pos_tol=5e-4, rot_tol=5e-3, return_error=True)
        error = array(error)
        if np.linalg.norm(error[:3]) > .002 or np.linalg.norm(error[3:]) > .03:
            qpos, error = robot.inverse_kinematics(
                link=self.hands[side], pos=tcp, quat=self.hand_quat[side], local_point=[0, 0, .1029],
                init_qpos=qpos, max_samples=40, max_solver_iters=200,
                pos_tol=2e-4, rot_tol=2e-3, return_error=True)
            error = array(error)
        if np.linalg.norm(error[:3]) > .002 or np.linalg.norm(error[3:]) > .03:
            raise RuntimeError(f"{side} IK failed for {tcp}: error {error}")
        qpos = array(qpos)
        qpos[7:] = self.grip[side]
        return qpos

    def move_joint(self, target_tcp, seconds):
        start = {side: array(self.robots[side].get_qpos()) for side in target_tcp}
        target = {side: self.solve(side, tcp) for side, tcp in target_tcp.items()}
        for i in range(1, round(seconds / self.dt) + 1):
            u = i / round(seconds / self.dt)
            s = u ** 3 * (10 - 15 * u + 6 * u ** 2)
            for side in target_tcp:
                q = start[side] + s * (target[side] - start[side])
                q[7:] = self.grip[side]
                self.robots[side].control_dofs_position(q)
            self.step()
        self.wait(.15)

    def move_cartesian(self, target_tcp, seconds):
        start = {side: self.tcp(side) for side in target_tcp}
        n = round(seconds / self.dt)
        # IK targets are refreshed at 50 Hz; joint controllers run every simulation step.
        refresh = max(1, round(.02 / self.dt))
        for i in range(1, n + 1):
            if i == 1 or i % refresh == 0 or i == n:
                u = i / n
                s = u ** 3 * (10 - 15 * u + 6 * u ** 2)
                for side in target_tcp:
                    q = self.solve(side, start[side] + s * (target_tcp[side] - start[side]))
                    self.robots[side].control_dofs_position(q)
            self.step()
        self.wait(.2)
        errors = {side: float(np.linalg.norm(self.tcp(side) - target_tcp[side])) for side in target_tcp}
        if max(errors.values()) > .01:
            raise RuntimeError(f"Robot failed to track Cartesian target: {errors}")

    def contact_candidates(self, side):
        points = array(self.objects["sloth"].get_particles_pos())
        candidates = []
        details = []
        for finger in self.fingers[side]:
            local = (points - array(finger.get_pos(relative=False))) @ rotation(array(finger.get_quat(relative=False)))
            # Main fingertip-pad collision box authored in panda.xml.
            d = np.abs(local - [.0, .0055, .0445]) - [.0085, .004, .0085]
            distance = np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(axis=1), 0)
            allowed = self.surface_indices[distance[self.surface_indices] <= .004]
            allowed = np.array([idx for idx in allowed if int(idx) not in self.used_particles], dtype=int)
            if len(allowed):
                # Keep the attachment local; at most four surface nodes per finger.
                allowed = allowed[np.argsort(distance[allowed])[:4]]
            candidates.append(allowed)
            details.append({"finger": finger.name, "nearest_pad_signed_distance_m": float(distance[self.surface_indices].min()),
                            "particle_indices": allowed.tolist(), "particle_pad_signed_distances_m": distance[allowed].tolist()})
        return candidates, details

    def fit_grasp(self, side):
        points = array(self.objects["sloth"].get_particles_pos())
        anchor = self.grasp_anchors[side]
        indices = self.surface_indices[(np.abs(points[self.surface_indices, 0] - anchor[0]) < .035)
                                       & (np.abs(points[self.surface_indices, 1] - anchor[1]) < .045)]
        pairs = []
        # Pair opposite surface nodes that both fit within the authored pad's x/z span.
        # Their midpoint centers the jaw opening; the distance gate is still checked
        # against current particles and actual finger poses during closure.
        for i in indices:
            for j in indices:
                delta = points[j] - points[i]
                if not .024 < delta[1] < .056 or abs(delta[0]) > .012 or abs(delta[2]) > .012:
                    continue
                center = (points[i] + points[j]) / 2
                if not self.table + .017 < center[2] < self.table + .036:
                    continue
                score = ((center[0] - anchor[0]) / .018) ** 2 + ((center[1] - anchor[1]) / .026) ** 2
                score += ((center[2] - (self.table + .020)) / .009) ** 2
                score += .15 * (abs(delta[0]) / .012 + abs(delta[2]) / .012)
                pairs.append((float(score), int(i), int(j), center, float(delta[1] / 2 + .0005)))
        if not pairs:
            raise RuntimeError(f"No opposing surface-node pair fits the {side} finger pads")
        _, i, j, center, jaw = min(pairs, key=lambda pair: pair[0])
        self.task.setdefault("grasp_fits", []).append({
            "side": side, "time_seconds": self.step_count * self.dt,
            "surface_node_pair": [i, j], "tcp_m": center.tolist(), "predicted_contact_finger_position_m": jaw})
        return center

    def align_grasps(self, sides):
        self.start_phase("align_grasp")
        updated = {side: self.fit_grasp(side) for side in sides}
        self.move_cartesian(updated, .35)
        self.grasp_points.update(updated)
        self.grasp_q.update({side: self.solve(side, point) for side, point in updated.items()})

    def close_grippers_once(self):
        self.start_phase("close_grippers")
        n = round(.9 / self.dt)
        initial_grip = {side: self.grip[side].copy() for side in self.robots}
        for i in range(1, n + 1):
            for side, robot in self.robots.items():
                if side in self.attached:
                    continue
                self.grip[side] = initial_grip[side] + (np.array([.007, .007]) - initial_grip[side]) * i / n
                q = array(robot.get_qpos())
                # Preserve the arm's target rather than integrating tracking error.
                q[:7] = self.grasp_q[side][:7]
                q[7:] = self.grip[side]
                robot.control_dofs_position(q)
            self.step()
            if i % 4:
                continue
            for side, robot in self.robots.items():
                if side in self.attached:
                    continue
                q = array(robot.get_qpos())
                candidates, details = self.contact_candidates(side)
                if q[7:].max() > .03 or not all(len(indices) for indices in candidates):
                    continue
                if np.linalg.norm(self.tcp(side) - self.grasp_points[side]) > .01:
                    continue
                for finger, indices in zip(self.fingers[side], candidates):
                    self.objects["sloth"].fix_particles_to_link(finger.idx, particles_idx_local=indices)
                    self.used_particles.update(int(idx) for idx in indices)
                self.grip[side] = q[7:].copy()
                q[:7] = self.grasp_q[side][:7]
                q[7:] = self.grip[side]
                robot.control_dofs_position(q)
                self.attached[side] = {"time_seconds": self.step_count * self.dt, "finger_joint_positions_m": q[7:].tolist(),
                                       "pads": details, "particle_count": sum(map(len, candidates))}
                print(f"Contact-gated grasp: {side}, {sum(map(len, candidates))} surface particles", flush=True)
                self.persist()
            if len(self.attached) == 2:
                break

    def close_grippers(self):
        for attempt in range(3):
            self.task["grasp_attempts"] = attempt + 1
            self.close_grippers_once()
            if len(self.attached) == 2:
                return
            pending = [side for side in self.robots if side not in self.attached]
            self.task.setdefault("grasp_retry_diagnostics", []).append({
                "time_seconds": self.step_count * self.dt,
                "pads": {side: self.contact_candidates(side)[1] for side in pending}})
            if attempt == 2:
                break
            self.start_phase("reopen_gripper")
            initial = {side: self.grip[side].copy() for side in pending}
            n = round(.3 / self.dt)
            for i in range(1, n + 1):
                for side in pending:
                    self.grip[side] = initial[side] + (.04 - initial[side]) * i / n
                    q = self.grasp_q[side].copy()
                    q[7:] = self.grip[side]
                    self.robots[side].control_dofs_position(q)
                self.step()
            self.align_grasps(pending)
        if len(self.attached) != 2:
            diagnostics = {side: self.contact_candidates(side)[1] for side in self.robots}
            self.task["failed_grasp_diagnostics"] = diagnostics
            raise RuntimeError(f"Both-finger contact gate failed: {diagnostics}")

    def execute(self):
        self.frame()
        self.next_frame_time = 1 / self.args.record_fps
        self.start_phase("settle")
        self.wait(.75)
        self.baseline = self.sample()
        self.task["settled_baseline"] = self.baseline
        points = array(self.objects["sloth"].get_particles_pos())
        # Lateral limbs leave the torso and garment unobstructed for later dressing.
        center_x = (points[:, 0].min() + points[:, 0].max()) / 2
        span_x = np.ptp(points[:, 0])
        self.grasp_points = {}
        self.grasp_anchors = {}
        for side, sign in (("left", -1), ("right", 1)):
            x = center_x + sign * span_x * .36
            self.grasp_anchors[side] = np.array([x, -.11, self.table + .02])
            self.grasp_points[side] = self.fit_grasp(side)
        self.task["planned_grasp_tcp_m"] = {side: point.tolist() for side, point in self.grasp_points.items()}
        self.save_stage("initial")
        for side in ("left", "right"):
            self.start_phase(side + "_approach")
            high = self.grasp_points[side] + [0, 0, .23]
            self.move_joint({side: high}, 1.2)
        self.start_phase("descend")
        self.move_cartesian(self.grasp_points, 1.4)
        self.grasp_q = {side: self.solve(side, point) for side, point in self.grasp_points.items()}
        self.align_grasps(list(self.robots))
        self.save_stage("pregrasp")
        self.close_grippers()
        self.start_phase("grasp_hold")
        self.wait(.3)
        self.save_stage("grasped")
        self.start_phase("lift")
        lifted = {side: self.tcp(side) + [0, 0, self.args.lift_height] for side in self.robots}
        self.move_cartesian(lifted, 2.)
        self.start_phase("airborne_hold")
        self.wait(self.args.hold_seconds)
        final = self.sample()
        hold = [sample for sample in self.samples if sample["phase"] == "airborne_hold"]
        min_clearance = min(sample["table_clearance_m"] for sample in hold)
        center_rise = final["sloth_center_m"][2] - self.baseline["sloth_center_m"][2]
        hold_drift = float(np.linalg.norm(np.array(hold[-1]["sloth_center_m"]) - hold[0]["sloth_center_m"]))
        checks = {
            "both_grippers_contact_gated": len(self.attached) == 2,
            "entire_sloth_clear_of_table": min_clearance > .08,
            "center_rose_at_least_10cm": center_rise > .1,
            "hold_center_drift_below_1cm": hold_drift < .01,
            "finite_particle_and_robot_states": True,
            "no_inter_arm_contacts": self.max_arm_contacts == 0,
            "no_extreme_mesh_stretch": max(sample["max_edge_stretch_ratio"] for sample in self.samples) < 1.5,
        }
        self.task["measurements"] = {"minimum_hold_table_clearance_m": min_clearance, "center_rise_m": center_rise,
                                     "hold_center_drift_m": hold_drift, "attached_particles": len(self.used_particles),
                                     "total_particles": len(points), "max_inter_arm_contacts": self.max_arm_contacts,
                                     "maximum_edge_stretch_ratio": max(sample["max_edge_stretch_ratio"] for sample in self.samples)}
        self.task["checks"] = checks
        self.task["success"] = all(checks.values())
        self.save_stage("lifted")
        self.persist()
        if not all(checks.values()):
            raise RuntimeError(f"Lift validation failed: {checks}; measurements {self.task['measurements']}")
        print(f"LIFT VERIFIED (constraint grasp): clearance={min_clearance:.3f} m, rise={center_rise:.3f} m, hold drift={hold_drift:.4f} m", flush=True)
        return self.step_count

    def close(self):
        self.video.close()
        self.log.close()


def run_lift(scene, robots, objects, cameras, config, report, run, args):
    experiment = LiftExperiment(scene, robots, objects, cameras, config, report, run, args)
    try:
        return experiment.execute()
    finally:
        if not report["lift_task"].get("success"):
            np.savez(run / "failure_particles.npz", **{
                name: array(entity.get_particles_pos()) for name, entity in objects.items()})
            report["lift_task"]["last_robot_qpos"] = {
                side: array(robot.get_qpos()).tolist() for side, robot in robots.items()}
        experiment.close()


def garment_openings(vertices, faces):
    """Return ordered, oriented rings from the actual garment boundary edges."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = edges[counts[inverse] == 1]
    following = {int(a): int(b) for a, b in boundary}
    if len(following) != len(boundary):
        raise ValueError("Garment boundary is not an oriented manifold")
    rings = []
    while following:
        first = min(following)
        ring = [first]
        nxt = following.pop(first)
        while nxt != first:
            ring.append(nxt)
            nxt = following.pop(nxt)
        rings.append(np.array(ring, dtype=int))
    if len(rings) != 4:
        raise ValueError("Dressing requires the original four garment openings")
    hem = max(rings, key=lambda ring: vertices[ring, 1].mean())
    collar = min(rings, key=lambda ring: vertices[ring, 1].mean())
    sleeves = [ring for ring in rings if ring is not hem and ring is not collar]
    sleeves.sort(key=lambda ring: vertices[ring, 0].mean())
    return {"hem": hem, "collar": collar, "left_sleeve": sleeves[0], "right_sleeve": sleeves[1]}


def restore_lift_pose(robots, objects, report, source):
    """Initialize a continuation from an actual recorded lift, before task stepping."""
    recorded = json.loads((source / "report.json").read_text())
    if recorded["status"] != "lift_verified_constraint":
        raise ValueError("Continuation requires a successfully verified lift")
    if recorded["asset_sha256"] != report["asset_sha256"]:
        raise ValueError("Continuation assets differ from the current scene")
    if recorded["genesis"]["materials"] != report["genesis"]["materials"]:
        raise ValueError("Continuation material parameters differ from the current scene")
    last = json.loads((source / "trajectory.jsonl").read_text().splitlines()[-1])
    with np.load(source / "final_particles.npz", allow_pickle=False) as data:
        positions = {name: data[name].copy() for name in objects}
    for name, values in positions.items():
        if values.shape != (objects[name].n_particles, 3) or not np.isfinite(values).all():
            raise ValueError(f"Invalid recorded particle state for {name}")
    for side, robot in robots.items():
        qpos = np.asarray(last["robot_qpos"][side])
        if qpos.shape != (robot.n_qs,) or not np.isfinite(qpos).all():
            raise ValueError(f"Invalid recorded joint state for {side}")
    for name, values in positions.items():
        objects[name].set_particles_pos(values)
        objects[name].set_particles_vel(0.)
    for side, robot in robots.items():
        qpos = np.asarray(last["robot_qpos"][side])
        robot.set_dofs_position(qpos)
        robot.set_dofs_velocity(np.zeros(robot.n_dofs))
        robot.control_dofs_position(qpos)
    report["continuation_initialization"] = {
        "source_run": str(source), "source_simulation_seconds": recorded["simulation_seconds"],
        "initialization_position_setters_used": True, "initial_velocities_reset_to_zero": True,
        "material_and_asset_match_verified": True, "recorded_grasps": recorded["lift_task"]["attachments"],
        "recorded_lift_checks": recorded["lift_task"]["checks"],
    }


def restore_dressing_pose(robots, objects, report, snapshot):
    """Restore a recorded finite supported state, including the saved velocities."""
    if snapshot.name not in ("state-supported.npz", "state-opened-hem.npz"):
        raise ValueError("This continuation requires a recorded supported or opened-hem stage")
    opened = snapshot.name == "state-opened-hem.npz"
    recorded = json.loads((snapshot.parent / "report.json").read_text())
    if recorded["asset_sha256"] != report["asset_sha256"] or recorded["genesis"]["materials"] != report["genesis"]["materials"]:
        raise ValueError("Supported-state assets or materials differ from the current scene")
    bindings = recorded["lift_task"]["attachments"]
    if set(bindings) != {"left"}:
        raise ValueError("Supported continuation requires the unchanged left-arm grasp")
    with np.load(snapshot, allow_pickle=False) as data:
        values = {name: data[name].copy() for name in data.files}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("Supported-state snapshot is non-finite")
    for name, entity in objects.items():
        entity.set_particles_pos(values[name + "_positions"])
        entity.set_particles_vel(values[name + "_velocities"])
    for side, robot in robots.items():
        robot.set_dofs_position(values[side + "_qpos"])
        robot.set_dofs_velocity(values[side + "_velocities"])
        robot.control_dofs_position(values[side + "_qpos"])
    report["continuation_initialization"] = {
        "source_run": str(snapshot.parent), "snapshot": str(snapshot), "resume_phase": "opened" if opened else "supported",
        "initialization_position_setters_used": True, "initial_velocities_reset_to_zero": False,
        "material_and_asset_match_verified": True, "recorded_grasps": bindings,
        "recorded_lift_checks": recorded["lift_task"]["checks"],
    }
    if opened:
        cloth_grasps = recorded["dressing_task"]["cloth_grasps"]
        if len(cloth_grasps) != 1 or cloth_grasps[0]["side"] != "right":
            raise ValueError("Opened-hem continuation requires the recorded right-hand cloth grasp")
        report["continuation_initialization"]["recorded_cloth_grasp"] = cloth_grasps[0]


def restore_dressing_snapshot(robots, objects, report, snapshot):
    metadata = json.loads(snapshot.with_suffix(".json").read_text())
    phases = {"state-rear-grasp": "rear", "state-head-down": "head-down",
              "state-cloth-grasp": "cloth-grasp",
              "state-opened-hem": "opened-rear", "state-inserted": "inserted"}
    if snapshot.stem not in phases:
        raise ValueError("This dressing stage is not supported for continuation")
    if metadata["asset_sha256"] != report["asset_sha256"] or metadata["materials"] != report["genesis"]["materials"]:
        raise ValueError("Dressing checkpoint assets or material properties differ")
    if set(metadata["sloth_grasps"]) != {"left"}:
        raise ValueError("Dressing checkpoint requires its recorded left-arm sloth grasp")
    with np.load(snapshot, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("Dressing checkpoint is non-finite")
    for name, entity in objects.items():
        entity.set_particles_pos(values[name + "_positions"])
        entity.set_particles_vel(values[name + "_velocities"])
    for side, robot in robots.items():
        robot.set_dofs_position(values[side + "_qpos"])
        robot.set_dofs_velocity(values[side + "_velocities"])
        robot.control_dofs_position(values[side + "_qpos"])
    initialization = report["continuation_initialization"] = {
        "source_run": str(snapshot.parent), "snapshot": str(snapshot), "resume_phase": phases[snapshot.stem],
        "initialization_position_setters_used": True, "initial_velocities_reset_to_zero": False,
        "material_and_asset_match_verified": True, "recorded_grasps": metadata["sloth_grasps"],
        "recorded_lift_checks": metadata["lift_checks"],
        "recorded_hand_quaternions": metadata["hand_quaternions"],
    }
    if metadata["cloth_bindings"]:
        if set(metadata["cloth_bindings"]) != {"right"}:
            raise ValueError("Dressing checkpoint supports one right-hand cloth grasp")
        binding = [grasp for grasp in metadata["cloth_grasp_history"] if grasp["side"] == "right"][-1]
        expected = set(np.concatenate(metadata["cloth_bindings"]["right"]).tolist())
        if expected != set(binding["attached_particle_indices"]):
            raise ValueError("Saved cloth nodes do not match the contact-gated grasp proof")
        initialization["recorded_cloth_grasp"] = binding


def winding_number(points, vertices, faces):
    """Generalized winding of a closed triangle mesh, without ray casting dependencies."""
    triangles = vertices[faces]
    result = []
    for batch in np.array_split(points, max(1, len(points) // 32)):
        a, b, c = (triangles[None, :, i] - batch[:, None] for i in range(3))
        la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
        numerator = np.einsum("...i,...i->...", a, np.cross(b, c))
        denominator = la * lb * lc + np.einsum("...i,...i->...", a, b) * lc
        denominator += np.einsum("...i,...i->...", b, c) * la + np.einsum("...i,...i->...", c, a) * lb
        result.extend(np.sum(2 * np.arctan2(numerator, denominator), axis=1) / (4 * np.pi))
    return np.asarray(result)


def threads_ring(inside, outside, ring):
    """Require an anatomical segment to cross the interior of an actual opening."""
    from matplotlib.path import Path

    center = ring.mean(0)
    _, singular, basis = np.linalg.svd(ring - center, full_matrices=False)
    normal = basis[2]
    signed = np.array([np.dot(inside - center, normal), np.dot(outside - center, normal)])
    denominator = signed[0] - signed[1]
    t = signed[0] / denominator if abs(denominator) > 1e-9 else -1.
    crossing = inside + t * (outside - inside)
    polygon = (ring - center) @ basis[:2].T
    projected = (crossing - center) @ basis[:2].T
    area = .5 * abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
                     - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1)))
    success = (0 < t < 1 and area > 1e-4 and Path(polygon).contains_point(projected))
    return {"threaded": bool(success), "segment_fraction": float(t), "opening_area_m2": float(area),
            "opening_center_m": center.tolist(), "endpoint_plane_distances_m": signed.tolist(),
            "planarity_residual_m": float(singular[-1] / np.sqrt(len(ring)))}


class DressingExperiment(LiftExperiment):
    """Lift first, then insert through a robot-opened hem with the table as support."""

    def __init__(self, *arguments):
        super().__init__(*arguments)
        config = arguments[4]
        self.cloth_bindings = {}
        self.peak_cloth_stretch = 1.
        with np.load(self.run / "assets" / "physics_meshes.npz") as physics:
            self.cloth_rest = physics["garment_points"].copy()
            self.cloth_faces = physics["garment_faces"].copy()
        self.rings = garment_openings(self.cloth_rest, self.cloth_faces)
        self.cloth_edges = np.unique(np.sort(np.concatenate([
            self.cloth_faces[:, [0, 1]], self.cloth_faces[:, [1, 2]], self.cloth_faces[:, [2, 0]]]), axis=1), axis=0)
        self.cloth_rest_lengths = np.linalg.norm(
            self.cloth_rest[self.cloth_edges[:, 0]] - self.cloth_rest[self.cloth_edges[:, 1]], axis=1)
        import trimesh
        normals = trimesh.Trimesh(self.cloth_rest, self.cloth_faces, process=False).vertex_normals
        front = normals[:, 2] > .15
        self.hem_front = np.flatnonzero(front & (self.cloth_rest[:, 1] > self.cloth_rest[self.rings["hem"], 1].min() - .025))
        self.dress = self.report["dressing_task"] = {
            "controller": "scripted robot IK; simulator geometry selects grasps and checks garment openings",
            "grasp_model": "local no-slip gripper attachments activated by both finger-pad contacts",
            "sloth_material": self.report["sloth_soft_body"],
            "garment_material": self.report["genesis"]["materials"]["garment"],
            "garment_model": "PBD.Cloth", "object_position_setters_used": False,
            "external_object_support_constraints": False, "success": False,
            "garment_opening_particle_indices": {name: ring.tolist() for name, ring in self.rings.items()},
            "cloth_grasps": [], "phases": [],
        }
        self.report["limitations"][0] = "Scripted dressing attempt. Completion requires torso enclosure and head/arm threading checks; render coverage alone is not success."
        self.in_dressing = False
        self.dress.update(object_position_setters_used=False,
                          object_target_pose_setters_used=False,
                          tabletop_friction=self.report["rigid_pbd_contact"],
                          surface_contact_solver=dict(self.report["surface_contact"]))

    def after_physics_step(self):
        self.dress["surface_contact_solver"]["peak_constraints"] = (
            self.objects["sloth"].solver.surface_contacts.peak_constraints)

    def start_phase(self, name):
        if self.in_dressing:
            self.dress["phases"].append({"name": name, "start_seconds": self.step_count * self.dt})
        super().start_phase(name)

    def save_stage(self, name):
        super().save_stage(name)
        np.savez(self.run / f"state-{name}.npz", **{
            **{f"{label}_positions": array(entity.get_particles_pos()) for label, entity in self.objects.items()},
            **{f"{label}_velocities": array(entity.get_particles_vel()) for label, entity in self.objects.items()},
            **{f"{side}_qpos": array(robot.get_qpos()) for side, robot in self.robots.items()},
            **{f"{side}_velocities": array(robot.get_dofs_velocity()) for side, robot in self.robots.items()},
        })
        (self.run / f"state-{name}.json").write_text(json.dumps({
            "asset_sha256": self.report["asset_sha256"], "materials": self.report["genesis"]["materials"],
            "phase": self.phase, "seconds": self.step_count * self.dt,
            "sloth_grasps": self.attached, "lift_checks": self.task.get("checks", {}),
            "cloth_bindings": {side: [indices.tolist() for indices in groups]
                               for side, groups in self.cloth_bindings.items()},
            "cloth_grasp_history": self.dress["cloth_grasps"], "hand_quaternions": self.hand_quat,
            "surface_contact_solver": self.dress["surface_contact_solver"],
        }, indent=2))

    def sample(self):
        sample = super().sample()
        points = array(self.objects["garment"].get_particles_pos())
        velocities = array(self.objects["garment"].get_particles_vel())
        if not np.isfinite(points).all() or not np.isfinite(velocities).all():
            raise RuntimeError("Garment has non-finite particle state")
        stretch = np.linalg.norm(points[self.cloth_edges[:, 0]] - points[self.cloth_edges[:, 1]], axis=1) / self.cloth_rest_lengths
        self.peak_cloth_stretch = max(self.peak_cloth_stretch, float(stretch.max()))
        self.dress["latest_cloth_state"] = {
            "center_m": points.mean(0).tolist(), "max_edge_stretch_ratio": float(stretch.max()),
            "table_clearance_m": float(points[:, 2].min() - self.table),
            "max_speed_m_s": float(np.linalg.norm(velocities, axis=1).max()),
        }
        return sample

    def open_hand(self, side, seconds=.35):
        if side in self.attached:
            for pad in self.attached.pop(side)["pads"]:
                indices = pad["particle_indices"]
                self.objects["sloth"].release_particle(particles_idx_local=indices)
                self.used_particles.difference_update(indices)
        if side in self.cloth_bindings:
            for indices in self.cloth_bindings.pop(side):
                self.objects["garment"].release_particle(particles_idx_local=indices)
        if abs(rotation(self.hand_quat[side])[2, 2]) < .5 and self.tcp(side)[2] < self.table + .075:
            self.move_cartesian({side: self.tcp(side) + [0, 0, .06]}, .6)
        initial = self.grip[side].copy()
        q = array(self.robots[side].get_qpos())
        n = round(seconds / self.dt)
        for i in range(1, n + 1):
            self.grip[side] = initial + (.04 - initial) * i / n
            q[7:] = self.grip[side]
            self.robots[side].control_dofs_position(q)
            self.step()

    def cloth_pad_candidates(self, side, eligible):
        points = array(self.objects["garment"].get_particles_pos())
        groups, diagnostics = [], []
        for finger in self.fingers[side]:
            local = (points - array(finger.get_pos(relative=False))) @ rotation(array(finger.get_quat(relative=False)))
            d = np.abs(local - [0, .0055, .0445]) - [.0085, .004, .0085]
            distance = np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(axis=1), 0)
            nearby = eligible[distance[eligible] <= .004]
            nearby = nearby[np.argsort(distance[nearby])[:4]]
            groups.append(nearby)
            diagnostics.append({"finger": finger.name, "particle_indices": nearby.tolist(),
                                "particle_pad_signed_distances_m": distance[nearby].tolist(),
                                "nearest_eligible_distance_m": float(distance[eligible].min())})
        return groups, diagnostics

    def grasp_cloth(self, side, eligible, anchor=None, phase="grasp_hem"):
        self.start_phase(phase)
        # The Panda's horizontal palm/finger geometry intersects the tabletop here.
        # Approach vertically with narrow jaws and stop as soon as both pads meet the cloth.
        self.hand_quat[side] = [0, 1, 0, 0]
        points = array(self.objects["garment"].get_particles_pos())
        anchor = points[self.rings["hem"]].mean(0) if anchor is None else np.asarray(anchor)
        score = np.linalg.norm(points[eligible] - anchor, axis=1) ** 2
        # A mean of coarse mesh nodes can be centimetres from any simulated particle.
        closest = np.array([eligible[np.argmin(score)]])
        target = points[closest].mean(0)
        target[2] = max(self.table + .018, target[2] + .004)
        self.move_joint({side: target + [0, 0, .18]}, 1.)
        self.grip[side] = np.array([.001, .001])
        q = array(self.robots[side].get_qpos()); q[7:] = self.grip[side]
        self.robots[side].control_dofs_position(q)
        self.wait(.25)
        start = self.tcp(side)
        n = round(1.2 / self.dt)
        details = []
        for i in range(1, n + round(.25 / self.dt) + 1):
            if i == 1 or i % max(1, round(.02 / self.dt)) == 0:
                current_patch = array(self.objects["garment"].get_particles_pos())[closest].mean(0)
                target[:2] += np.clip(current_patch[:2] - target[:2], -.002, .002)
                u = min(i / n, 1.); s = u ** 3 * (10 - 15 * u + 6 * u ** 2)
                arm_target = self.solve(side, start + s * (target - start))
            command = arm_target.copy(); command[7:] = self.grip[side]
            self.robots[side].control_dofs_position(command)
            self.step()
            groups, details = self.cloth_pad_candidates(side, eligible)
            q = array(self.robots[side].get_qpos())
            if q[7:].max() < .004 and all(len(group) for group in groups):
                # Both pads may contact the same thin-sheet vertex. Bind its union once to the
                # actual hand, after both independent pad-distance checks have passed.
                indices = np.unique(np.concatenate(groups))
                self.objects["garment"].fix_particles_to_link(self.hands[side].idx, particles_idx_local=indices)
                self.cloth_bindings[side] = [indices]
                self.grip[side] = q[7:].copy()
                command = q.copy()
                command[7:] = self.grip[side]
                self.robots[side].control_dofs_position(command)
                self.dress["cloth_grasps"].append({"side": side, "time_seconds": self.step_count * self.dt,
                                                  "target_m": target.tolist(), "pads": details,
                                                  "attachment_link": self.hands[side].name,
                                                  "attached_particle_indices": indices.tolist()})
                print(f"Contact-gated garment grasp: {side}, {len(indices)} particles", flush=True)
                self.wait(.15)
                self.persist()
                return
        self.dress["failed_cloth_grasp"] = details
        self.save_stage("failed-cloth-grasp")
        raise RuntimeError(f"Both garment finger pads failed the contact gate: {details}")

    def arrange_openings(self):
        """Pull the actual collar/cuffs toward the corresponding anatomical segment."""
        x = self.rest[:, 0] + .18; y = self.rest[:, 1]
        segments = {
            "collar": ((np.abs(x) < .025) & (y > -.13) & (y < -.09), (np.abs(x) < .028) & (y < -.183)),
            "right_sleeve": ((x > .035) & (x < .06) & (y < -.07) & (y > -.14), x > .112),
            "left_sleeve": ((x < -.035) & (x > -.06) & (y < -.07) & (y > -.14), x < -.112),
        }
        self.evaluate()
        if self.dress["measurements"]["torso_enclosure_fraction"] < .4:
            return
        for name, (inside, outside) in segments.items():
            body = array(self.objects["sloth"].get_particles_pos())
            cloth = array(self.objects["garment"].get_particles_pos())
            ring = self.rings[name]
            test = threads_ring(body[inside].mean(0), body[outside].mean(0), cloth[ring])
            if test["threaded"]:
                continue
            neighbor_faces = self.cloth_faces[np.isin(self.cloth_faces, ring).any(1)]
            candidates = np.unique(neighbor_faces)
            self.grasp_cloth("right", candidates, anchor=cloth[ring].mean(0), phase="grasp_" + name)
            self.start_phase("thread_" + name)
            body = array(self.objects["sloth"].get_particles_pos())
            cloth = array(self.objects["garment"].get_particles_pos())
            target = .35 * body[inside].mean(0) + .65 * body[outside].mean(0)
            displacement = target - cloth[ring].mean(0)
            if np.linalg.norm(displacement) > .16:
                raise RuntimeError(f"{name} is too far from its anatomical segment for a local pull")
            self.move_cartesian({"right": self.tcp("right") + displacement}, 1.5)
            self.open_hand("right")
            self.move_cartesian({"right": self.tcp("right") + [0, 0, .15]}, .7)
            self.wait(.3)
            self.save_stage("arranged-" + name)

    def grasp_rear_feet(self):
        side = "left"
        self.start_phase("regrasp_rear_feet")
        self.open_hand(side)
        self.move_cartesian({side: self.tcp(side) + [0, 0, .16]}, .8)
        self.wait(.4)
        eligible = self.surface_indices[self.rest[self.surface_indices, 1] > -.035]
        points = array(self.objects["sloth"].get_particles_pos())
        center_x = points[eligible, 0].mean()
        rest_center_x = (self.rest[:, 0].min() + self.rest[:, 0].max()) / 2
        left_rear = eligible[self.rest[eligible, 0] < rest_center_x - .025]
        right_rear = eligible[self.rest[eligible, 0] > rest_center_x + .025]
        pairs = []
        for i in left_rear:
            for j in right_rear:
                delta = points[j] - points[i]
                center = (points[i] + points[j]) / 2
                if (.04 < np.linalg.norm(delta) < .087 and abs(delta[1]) < .035 and abs(delta[2]) < .023
                        and center[2] > self.table + .018):
                    score = ((center[0] - center_x) / .04) ** 2 + ((center[1] - points[eligible, 1].max()) / .05) ** 2
                    score += .2 * (abs(delta[1]) + abs(delta[2])) / .012
                    pairs.append((score, int(i), int(j), center))
        if not pairs:
            self.save_stage("failed-rear-grasp-plan")
            raise RuntimeError("No opposed rear-foot surface nodes fit the real finger pads")
        _, first, second, target = min(pairs, key=lambda pair: pair[0])
        jaw_axis = points[second] - points[first]
        jaw_axis /= np.linalg.norm(jaw_axis)
        tool_axis = np.array([0., 0., -1.])
        tool_axis -= jaw_axis * np.dot(tool_axis, jaw_axis)
        tool_axis /= np.linalg.norm(tool_axis)
        matrix = np.column_stack([np.cross(jaw_axis, tool_axis), jaw_axis, tool_axis])
        self.hand_quat[side] = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]].tolist()
        self.dress["rear_grasp_plan"] = {"surface_node_pair": [first, second], "tcp_m": target.tolist()}
        self.persist()
        self.move_joint({side: target + [0, 0, .15]}, .8)
        self.move_cartesian({side: target}, 1.)
        arm_q = self.solve(side, target)
        initial = self.grip[side].copy()
        for i in range(1, round(.9 / self.dt) + 1):
            self.grip[side] = initial + (np.array([.008, .008]) - initial) * i / round(.9 / self.dt)
            command = arm_q.copy(); command[7:] = self.grip[side]
            self.robots[side].control_dofs_position(command)
            self.step()
            groups, details = self.contact_candidates(side)
            groups = [np.intersect1d(group, eligible) for group in groups]
            q = array(self.robots[side].get_qpos())
            on_both_legs = all(len(group) for group in groups) and any(np.isin(group, left_rear).any() for group in groups) and any(np.isin(group, right_rear).any() for group in groups)
            if q[7:].max() <= .04 and on_both_legs:
                for finger, group, detail in zip(self.fingers[side], groups, details):
                    self.objects["sloth"].fix_particles_to_link(finger.idx, particles_idx_local=group)
                    self.used_particles.update(group.tolist())
                    distances = dict(zip(detail["particle_indices"], detail["particle_pad_signed_distances_m"]))
                    detail["particle_indices"] = group.tolist()
                    detail["particle_pad_signed_distances_m"] = [distances[int(idx)] for idx in group]
                self.grip[side] = q[7:].copy()
                self.robots[side].control_dofs_position(q)
                self.attached[side] = {"time_seconds": self.step_count * self.dt,
                                       "finger_joint_positions_m": q[7:].tolist(), "pads": details,
                                       "particle_count": sum(map(len, groups)), "anatomical_region": "both rear legs"}
                self.wait(.2)
                self.save_stage("rear-grasp")
                return
        self.dress["failed_rear_grasp_contacts"] = details
        self.save_stage("failed-rear-grasp")
        raise RuntimeError("Rear-foot grasp did not meet both finger-pad contact gates")

    def rotate_hand(self, side, axis, angle, phase):
        self.start_phase(phase)
        initial = rotation(self.hand_quat[side])
        target = self.tcp(side).copy()
        n = round(1.5 / self.dt)
        for i in range(1, n + 1):
            if i == 1 or i % max(1, round(.02 / self.dt)) == 0 or i == n:
                u = i / n; s = u ** 3 * (10 - 15 * u + 6 * u ** 2)
                quat = Rotation.from_matrix(Rotation.from_rotvec(np.asarray(axis) * angle * s).as_matrix() @ initial).as_quat()
                self.hand_quat[side] = quat[[3, 0, 1, 2]].tolist()
                self.robots[side].control_dofs_position(self.solve(side, target))
            self.step()
        self.wait(.2)

    def orient_sloth_for_dressing(self):
        # Keep the palm above the two rear legs; gravity bends the body head-down.
        tool = rotation(self.hand_quat["left"])[:, 2]
        current_angle = np.arctan2(tool[1], -tool[2])
        angle = -current_angle
        if abs(angle) > .01:
            self.rotate_hand("left", [1, 0, 0], angle, "orient_sloth")
        self.save_stage("head-down")

    def align_garment_axis(self):
        body = array(self.objects["sloth"].get_particles_pos())
        cloth = array(self.objects["garment"].get_particles_pos())
        x = self.rest[:, 0] - (self.rest[:, 0].min() + self.rest[:, 0].max()) / 2
        arm_axis = (body[x > .112].mean(0) - body[x < -.112].mean(0))[:2]
        sleeve_axis = (cloth[self.rings["right_sleeve"]].mean(0) - cloth[self.rings["left_sleeve"]].mean(0))[:2]
        angle = np.arctan2(sleeve_axis[0] * arm_axis[1] - sleeve_axis[1] * arm_axis[0], np.dot(sleeve_axis, arm_axis))
        self.dress["garment_axis_alignment"] = {"yaw_rotation_rad": float(angle),
                "body_arm_axis_xy": arm_axis.tolist(), "garment_sleeve_axis_xy": sleeve_axis.tolist()}
        if abs(angle) > .08:
            self.rotate_hand("right", [0, 0, 1], float(angle), "orient_garment")
        self.wait(.4)

    def evaluate(self):
        sloth = array(self.objects["sloth"].get_particles_pos())
        cloth = array(self.objects["garment"].get_particles_pos())
        vertices = [*cloth]; caps = []
        for ring in self.rings.values():
            center_idx = len(vertices); vertices.append(cloth[ring].mean(0))
            caps.extend([[int(b), int(a), center_idx] for a, b in zip(ring, np.roll(ring, -1))])
        closed_faces = np.concatenate([self.cloth_faces, np.asarray(caps)])
        enclosure = np.abs(winding_number(sloth, np.asarray(vertices), closed_faces)) > .5
        x = self.rest[:, 0] - (self.rest[:, 0].min() + self.rest[:, 0].max()) / 2
        y = self.rest[:, 1]
        torso = (np.abs(x) < .045) & (y > -.158) & (y < -.06)
        segments = {
            "collar": ((np.abs(x) < .025) & (y > -.13) & (y < -.09), (np.abs(x) < .028) & (y < -.183)),
            "left_sleeve": ((x < -.035) & (x > -.06) & (y < -.07) & (y > -.14), x < -.112),
            "right_sleeve": ((x > .035) & (x < .06) & (y < -.07) & (y > -.14), x > .112),
        }
        threading = {name: threads_ring(sloth[inside].mean(0), sloth[outside].mean(0), cloth[self.rings[name]])
                     for name, (inside, outside) in segments.items()}
        ratio = float(enclosure[torso].mean())
        # Also sample the fabric faces/edges: particle-only checks can miss a face crossing the body.
        from dressing_scene import tetra_boundary
        with np.load(self.run / "assets" / "physics_meshes.npz") as archive:
            used, faces = tetra_boundary(self.rest, archive["sloth_tetra"])
        cloth_samples = np.concatenate([cloth, cloth[self.cloth_faces].mean(1), cloth[self.cloth_edges].mean(1)])
        interior = np.abs(winding_number(cloth_samples, sloth[used], faces)) > .5
        import trimesh
        mesh = trimesh.Trimesh(sloth[used], faces, process=False)
        _, distances, _ = trimesh.proximity.closest_point_naive(mesh, cloth_samples[interior]) if interior.any() else (
            None, np.array([]), None)
        deep = int(np.sum(distances > .003))
        checks = {"torso_enclosed_at_least_80_percent": ratio >= .8,
                  "head_through_collar": threading["collar"]["threaded"],
                  "left_arm_through_sleeve": threading["left_sleeve"]["threaded"],
                  "right_arm_through_sleeve": threading["right_sleeve"]["threaded"],
                  "no_deep_cloth_surface_penetrations": deep == 0,
                  "no_extreme_cloth_stretch": self.peak_cloth_stretch < 1.5,
                  "no_extreme_sloth_stretch": max(sample["max_edge_stretch_ratio"] for sample in self.samples) < 1.5,
                  "sloth_volume_within_20_percent": all(.8 <= sample["sloth_volume_ratio"] <= 1.2 for sample in self.samples),
                  "no_large_tetra_inversions": max(sample["inverted_tetra_rest_volume_fraction"] for sample in self.samples) < .01,
                  "garment_released": not self.cloth_bindings,
                  "sloth_released": not self.attached,
                  "finite_particle_and_robot_states": True,
                  "no_inter_arm_contacts": self.max_arm_contacts == 0}
        self.dress.update(checks=checks, measurements={"torso_enclosure_fraction": ratio,
                          "opening_threading": threading, "deeply_penetrating_cloth_surface_samples": deep,
                          "maximum_cloth_penetration_m": float(distances.max(initial=0)),
                          "maximum_cloth_edge_stretch_ratio": self.peak_cloth_stretch},
                          success=all(checks.values()))
        print(f"Dressing checks: {checks}", flush=True)
        self.persist()
        return all(checks.values())

    def execute(self):
        resume_phase = self.report.get("continuation_initialization", {}).get("resume_phase")
        rear_continuation = resume_phase in ("rear", "head-down", "cloth-grasp", "opened-rear", "inserted")
        supported_continuation = resume_phase in ("supported", "opened") or rear_continuation
        opened_continuation = resume_phase in ("opened", "opened-rear", "inserted")
        if "continuation_initialization" in self.report:
            initialization = self.report["continuation_initialization"]
            self.hand_quat.update(initialization.get("recorded_hand_quaternions", {}))
            self.attached.update(initialization["recorded_grasps"])
            for side, binding in self.attached.items():
                self.grip[side] = array(self.robots[side].get_qpos())[7:]
                for finger, pad in zip(self.fingers[side], binding["pads"]):
                    indices = pad["particle_indices"]
                    self.objects["sloth"].fix_particles_to_link(finger.idx, particles_idx_local=indices)
                    self.used_particles.update(indices)
            if "recorded_cloth_grasp" in initialization:
                binding = initialization["recorded_cloth_grasp"]
                side = binding["side"]
                indices = np.asarray(binding["attached_particle_indices"])
                self.objects["garment"].fix_particles_to_link(self.hands[side].idx, particles_idx_local=indices)
                self.cloth_bindings[side] = [indices]
                self.grip[side] = array(self.robots[side].get_qpos())[7:]
                self.dress["cloth_grasps"].append({**binding, "restored_from": initialization["source_run"]})
            self.task.update(success=True, continued_from=initialization["source_run"],
                             checks=initialization["recorded_lift_checks"])
            self.frame()
            self.next_frame_time = 1 / self.args.record_fps
            self.start_phase("open_hem" if opened_continuation else "support_sloth" if supported_continuation else "airborne_hold")
            self.wait(.2 if supported_continuation else .4)
            self.save_stage("continued-supported" if supported_continuation else "continued-lift")
        else:
            super().execute()
        self.in_dressing = True
        self.cameras["lift"].set_pose(pos=(.5, -.9, 1.35), lookat=(.03, 0, .84))
        if not supported_continuation:
            self.start_phase("transfer_sloth")
            # Keep the wrists far enough from the bases to respect the elbow's flexion limit.
            transfer = {side: self.tcp(side) + [.21, .25, .01] for side in self.robots}
            self.move_cartesian(transfer, 2.5)
            self.start_phase("support_sloth")
            points = array(self.objects["sloth"].get_particles_pos())
            lower = self.table + .003 - points[:, 2].min()
            self.move_cartesian({side: self.tcp(side) + [0, 0, lower] for side in self.robots}, 1.5)
            self.open_hand("right")
            self.move_cartesian({"right": self.tcp("right") + [0, 0, .17]}, .8)
            self.save_stage("supported")
        if not rear_continuation:
            self.grasp_rear_feet()
        if resume_phase not in ("head-down", "cloth-grasp", "opened-rear", "inserted"):
            self.start_phase("park_right_arm")
            self.move_cartesian({"right": np.array([.33, -.14, 1.07])}, 1.)
            self.start_phase("raise_rear_feet")
            self.move_cartesian({"left": self.tcp("left") + [0, 0, .31]}, 1.8)
        self.orient_sloth_for_dressing()
        if not opened_continuation:
            if "right" not in self.cloth_bindings:
                self.grasp_cloth("right", self.hem_front)
                self.save_stage("cloth-grasp")
            self.start_phase("open_hem")
            start = self.tcp("right").copy()
            self.move_cartesian({"right": np.array([.23, start[1], self.table + .09])}, .8)
            self.rotate_hand("right", [0, 1, 0], np.pi / 2, "orient_garment")
            self.start_phase("open_hem")
            self.move_cartesian({"right": start + [-.15, 0, .30]}, 2.2)
            self.align_garment_axis()
            self.save_stage("opened-hem")
        if resume_phase != "inserted":
            self.start_phase("align_to_hem")
            points = array(self.objects["sloth"].get_particles_pos())
            cloth = array(self.objects["garment"].get_particles_pos())
            head = points[(np.abs(self.rest[:, 0] + .18) < .028) & (self.rest[:, 1] < -.183)].mean(0)
            opening = cloth[self.rings["hem"]].mean(0)
            self.move_cartesian({"left": self.tcp("left") + opening + [0, 0, .06] - head}, 1.4)
            self.save_stage("aligned-hem")
            self.start_phase("insert_sloth")
            cloth = array(self.objects["garment"].get_particles_pos())
            head = array(self.objects["sloth"].get_particles_pos())[(np.abs(self.rest[:, 0] + .18) < .028)
                                                                  & (self.rest[:, 1] < -.183)].mean(0)
            collar = cloth[self.rings["collar"]].mean(0)
            axis = cloth[self.rings["hem"]].mean(0) - collar
            axis /= np.linalg.norm(axis)
            target_head = collar - .018 * axis
            self.dress["insertion_plan"] = {"head_start_m": head.tolist(), "target_head_m": target_head.tolist(),
                                             "collar_center_m": collar.tolist(), "garment_axis": axis.tolist()}
            self.persist()
            self.move_cartesian({"left": self.tcp("left") + target_head - head}, 3.)
            self.save_stage("inserted")
        self.start_phase("settle_garment")
        self.move_cartesian({"right": self.tcp("right") + [0, .015, -.04]}, 1.5)
        self.open_hand("right")
        self.move_cartesian({"right": self.tcp("right") + [0, 0, .20]}, 1.)
        self.arrange_openings()
        self.start_phase("release_sloth")
        self.open_hand("left")
        self.move_cartesian({"left": self.tcp("left") + [0, 0, .22]}, 1.)
        self.start_phase("dressed_hold")
        self.wait(1.5)
        self.sample()
        self.save_stage("dressed")
        np.savez(self.run / "dressing_particles.npz", **{name: array(entity.get_particles_pos())
                                                       for name, entity in self.objects.items()})
        if not self.evaluate():
            raise RuntimeError("Dressing is incomplete: garment enclosure or opening-threading checks failed")
        return self.step_count


def run_dressing(scene, robots, objects, cameras, config, report, run, args):
    experiment = DressingExperiment(scene, robots, objects, cameras, config, report, run, args)
    try:
        return experiment.execute()
    finally:
        if not report["dressing_task"].get("success"):
            np.savez(run / "failure_particles.npz", **{name: array(entity.get_particles_pos())
                                                       for name, entity in objects.items()})
            report["dressing_task"]["last_robot_qpos"] = {
                side: array(robot.get_qpos()).tolist() for side, robot in robots.items()}
        experiment.close()
