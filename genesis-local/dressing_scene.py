"""Load the bunnylab dressing assets/layout into the installed Genesis environment.

The default task settles the scene. --task lift-sloth runs a scripted dual-arm
lift with contact-gated PBD grasp constraints. Route A uses this scene through
the bunnylab CLI's Genesis bridge. Source assets are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import tomllib
import traceback
from datetime import datetime

import fast_simplification
import numpy as np
import trimesh
from PIL import Image
from genesis_mesh_adapter import (existing_pbd_mesh, simulation_connectivity, install_rigid_pbd_friction,
                                  install_cotton_solver, apply_cotton_parameters, install_surface_contacts)

HERE = Path(__file__).resolve().parent


def topology(vertices, faces, allow_nonmanifold=False):
    edges, counts = np.unique(np.sort(np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
        axis=0, return_counts=True)
    if np.any(counts > 2) and not allow_nonmanifold:
        raise ValueError("Mesh has non-manifold edges")

    def components(edges, indices):
        adjacency = {int(i): set() for i in indices}
        for a, b in edges:
            adjacency[int(a)].add(int(b))
            adjacency[int(b)].add(int(a))
        remaining = set(adjacency)
        count = 0
        while remaining:
            stack = [remaining.pop()]
            count += 1
            while stack:
                for j in adjacency[stack.pop()]:
                    if j in remaining:
                        remaining.remove(j)
                        stack.append(j)
        return count

    boundary = edges[counts == 1]
    boundary_vertices, degrees = np.unique(boundary, return_counts=True)
    if len(degrees) and np.any(degrees != 2):
        raise ValueError("Mesh boundary is not a collection of closed loops")
    return {"vertices": len(vertices), "faces": len(faces),
            "components": components(edges, np.arange(len(vertices))),
            "boundary_loops": components(boundary, boundary_vertices),
            "nonmanifold_edges": int(np.count_nonzero(counts > 2))}


def require_same_topology(before, after):
    for key in ("components", "boundary_loops"):
        if before[key] != after[key]:
            raise ValueError(f"Mesh topology changed: {before} -> {after}")


def read_obj(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        tokens = line.split()
        if tokens and tokens[0] == "v":
            vertices.append([float(v) for v in tokens[1:4]])
        elif tokens and tokens[0] == "f":
            polygon = [int(t.split("/")[0]) for t in tokens[1:]]
            polygon = [v - 1 if v > 0 else len(vertices) + v for v in polygon]
            faces.extend([[polygon[0], polygon[i], polygon[i + 1]]
                          for i in range(1, len(polygon) - 1)])
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def tetra_boundary(points, tetra):
    """Extract exterior triangles, oriented away from each tetra's opposite vertex."""
    faces = np.concatenate([tetra[:, [1, 2, 3]], tetra[:, [0, 3, 2]],
                            tetra[:, [0, 1, 3]], tetra[:, [0, 2, 1]]])
    opposite = np.concatenate([tetra[:, 0], tetra[:, 1], tetra[:, 2], tetra[:, 3]])
    _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Sloth tetra mesh has non-manifold faces")
    exterior = counts[inverse] == 1
    faces, opposite = faces[exterior].copy(), opposite[exterior]
    tri = points[faces]
    inward = np.einsum("ij,ij->i", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                       points[opposite] - tri[:, 0]) > 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    used, inverse = np.unique(faces, return_inverse=True)
    return used, inverse.reshape(-1, 3)


def write_obj(path, vertices, faces, uv=None):
    with path.open("w") as stream:
        stream.write("# Derived from bunnylab assets; original files are unchanged.\n")
        for vertex in vertices:
            stream.write("v " + " ".join(f"{x:.12g}" for x in vertex) + "\n")
        if uv is not None:
            for coord in uv:
                stream.write("vt " + " ".join(f"{x:.12g}" for x in coord) + "\n")
        for face in faces + 1:
            entries = [f"{i}/{i}" if uv is not None else str(i) for i in face]
            stream.write("f " + " ".join(entries) + "\n")


def prepare_assets(project, run):
    config_path = project / "config.toml"
    config = tomllib.loads(config_path.read_text())
    sc = config["scene"]
    paths = {key: (project / config["assets"][key]).resolve()
             for key in ("robot", "garment", "sloth", "sloth_texture")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required bunnylab asset is missing: {path}")

    with np.load(paths["sloth"], allow_pickle=False) as archive:
        points = archive["points"].copy()
        tetra = archive["tetra"].copy()
        uv = archive["texcoord"].copy()
    used, boundary = tetra_boundary(points, tetra)
    sloth = points[used].copy()
    sloth_topology = topology(sloth, boundary, allow_nonmanifold=True)
    sloth_mesh = trimesh.Trimesh(vertices=sloth, faces=boundary, process=False)
    if sloth_mesh.volume <= 0:
        raise ValueError("Sloth exterior must be outward-oriented")
    tetra_volume = float(np.abs(np.linalg.det(
        points[tetra[:, 1:]] - points[tetra[:, :1]])).sum() / 6)
    if not np.isclose(sloth_mesh.volume, tetra_volume, rtol=1e-5):
        raise ValueError("Sloth tetra volume does not match the extracted exterior")

    garment, faces = read_obj(paths["garment"])
    original_topology = topology(garment, faces)
    garment, faces = fast_simplification.simplify(
        garment, faces, target_count=sc["garment"]["target_faces"], agg=5.)
    garment_topology = topology(garment, faces)
    require_same_topology(original_topology, garment_topology)
    # Match build_scene() in bunnylab, including the two original fabric layers.
    garment -= (garment.max(0) + garment.min(0)) / 2
    garment *= sc["garment"]["width"] / np.ptp(garment[:, 1])
    garment = garment[:, [1, 0, 2]]
    garment[:, 0] *= -1
    garment_area = float(trimesh.Trimesh(vertices=garment, faces=faces, process=False).area)
    # Genesis cotton mass comes from measured areal density and actual two-sided
    # garment surface area; scene.garment.mass remains the MuJoCo legacy setting.
    cotton = sc.get("genesis", {}).get("garment_material")
    if cotton:
        density = cotton["areal_density_kg_m2"]
        if not np.isfinite(density) or density <= 0:
            raise ValueError("Cotton areal density must be positive and finite")
        sc["garment"]["mass"] = density * garment_area

    prepared = run / "assets"
    prepared.mkdir()
    for name, vertices, elements, texture_uv in (
        ("sloth", sloth, boundary, uv[used]), ("garment", garment, faces, None)
    ):
        params = sc[name]
        vertices[:, :2] += params["position_xy"]
        vertices[:, 2] += sc["table_height"] + params["radius"] + .004 - vertices[:, 2].min()
        write_obj(prepared / f"{name}.obj", vertices, elements, texture_uv)
    # Retain the original tetra/UV data for provenance and a future FEM adapter.
    np.savez(prepared / "sloth_source.npz", points=points, tetra=tetra, texcoord=uv)
    world_points = points.copy()
    world_points[:, :2] += sc["sloth"]["position_xy"]
    world_points[:, 2] += sc["table_height"] + sc["sloth"]["radius"] + .004 - points[:, 2].min()
    np.savez(prepared / "physics_meshes.npz", sloth_points=world_points, sloth_tetra=tetra,
             garment_points=garment, garment_faces=faces)

    report = {
        "status": "assets_prepared", "source_project": str(project),
        "source_config": str(config_path), "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "asset_paths": {key: str(path) for key, path in paths.items()},
        "asset_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
        "layout": {key: sc[key] for key in ("table_height", "table_half_size", "left_base", "right_base", "home", "cameras")},
        "source_materials": {name: sc[name] for name in ("sloth", "garment")},
        "geometry": {
            "sloth": {"source_vertices": len(points), "source_tetrahedra": len(tetra),
                      "surface": sloth_topology, "volume_m3": tetra_volume,
                      "bounds_m": [sloth.min(0).tolist(), sloth.max(0).tolist()]},
            "garment": {"original": original_topology, "prepared": garment_topology,
                        "area_m2": garment_area,
                        "bounds_m": [garment.min(0).tolist(), garment.max(0).tolist()]},
        },
        "limitations": [
            "Scene migration/settling only; no policy, grasping, or dressing success evaluation.",
            "A version-checked pre-build adapter for Genesis 1.4.3 bypasses PBD remeshing; it uses the same 551 sloth vertices/2549 tetrahedra and the same simplified MuJoCo garment mesh.",
            "The original sloth tetra exterior has one non-manifold edge; it is retained rather than repaired or substituted.",
            "Cotton areal density and bending rigidity use published measurements; isotropic membrane stiffness, drag and contact are approximations, not specimen calibration. Sloth PBD properties are uncalibrated soft-toy priors.",
            "MJCF finger tendon actuation is approximated by two joint position controllers in Genesis.",
            "The shared PBD particle diameter differs from the two MuJoCo vertex collision radii and garment thickness.",
            "PBD self-contact is particle-based and ignores pairs closer than a particle diameter in the rest pose; it is not continuous cloth self-contact.",
            "Upstream Genesis 1.4.3 omits legacy rigid/PBD friction; a scene-local version-checked adapter adds Coulomb friction to free particles, limited by the normal SDF projection. This does not validate physical contact pressure or frictional grasping.",
        ],
    }
    return config, paths, prepared, report


def build_scene(config, paths, prepared, report, args):
    import genesis as gs
    import torch
    from importlib.metadata import version

    requested = gs.cuda if args.backend == "cuda" else gs.cpu
    if args.backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the NVIDIA GPU. Run nvidia-smi in your local terminal; use --backend cpu explicitly for CPU.")
    gs.init(backend=requested, seed=config["experiment"]["seed"], precision="32", logging_level=logging.WARNING)
    if gs.backend != requested:
        raise RuntimeError("Genesis selected a different backend than requested")
    sc = config["scene"]
    dt = sc["timestep"] * args.substeps
    overview = sc["cameras"]["overview"]
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=args.substeps),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        pbd_options=gs.options.PBDOptions(
            particle_size=args.particle_size,
            max_stretch_solver_iterations=8, max_bending_solver_iterations=4,
            max_volume_solver_iterations=32,
            lower_bound=(-1.5, -1.5, 0), upper_bound=(1.5, 1.5, 2.5)),
        viewer_options=gs.options.ViewerOptions(
            res=(args.width, args.height), camera_pos=overview["position"],
            camera_lookat=overview["target"], camera_fov=overview["fovy"],
            refresh_rate=30),
        show_viewer=not args.headless,
    )
    cotton = sc.get("genesis", {}).get("garment_material")
    if cotton:
        install_cotton_solver(scene)
    report["rigid_pbd_contact"] = install_rigid_pbd_friction(scene)
    static_material = gs.materials.Rigid(needs_coup=True, friction=.7, coup_friction=.7)

    def box(name, pos, size, color):
        return scene.add_entity(name=name, morph=gs.morphs.Box(pos=pos, size=size, fixed=True),
                                material=static_material, surface=gs.surfaces.Default(color=color))

    scene.add_entity(name="floor", morph=gs.morphs.Plane(), material=static_material,
                     surface=gs.surfaces.Default(color=(.13, .17, .22, 1)))
    th, ts = sc["table_height"], np.array(sc["table_half_size"])
    box("table", (0, 0, th - ts[2]), tuple(2 * ts), (.66, .53, .37, 1))
    for i, x in enumerate((-ts[0] + .07, ts[0] - .07)):
        for j, y in enumerate((-ts[1] + .07, ts[1] - .07)):
            h = th - 2 * ts[2]
            box(f"table_leg_{i}_{j}", (x, y, h / 2), (.07, .07, h), (.2, .24, .28, 1))

    robots = {}
    for side, yaw in (("left", -.35), ("right", np.pi + .35)):
        bp = sc[side + "_base"]
        box(side + "_mount", (bp[0], bp[1], (bp[2] + th) / 2),
            (.2, .2, bp[2] - th), (.23, .28, .32, 1))
        robots[side] = scene.add_entity(
            name=side + "_franka", material=gs.materials.Rigid(
                needs_coup=True, coup_friction=.7,
                gravity_compensation=1.0 if getattr(args,"rpc",False) or args.task in ("lift-sloth", "dress-sloth") else 0.0),
            morph=gs.morphs.MJCF(file=str(paths["robot"]), pos=bp,
                                  quat=(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))),
        )

    sloth_density = sc["sloth"]["mass"] / report["geometry"]["sloth"]["volume_m3"]
    garment_density = sc["garment"]["mass"] / report["geometry"]["garment"]["area_m2"]
    sloth_soft_body = {
        "stretch_compliance": 5e-3, "volume_compliance": 1e-12,
        "stretch_relaxation": .3, "volume_relaxation": .3,
    }
    configured_material = sc.get("genesis", {}).get("sloth_material", {})
    unknown = set(configured_material) - set(sloth_soft_body)
    if unknown:
        raise ValueError(f"Unknown Genesis sloth material settings: {sorted(unknown)}")
    sloth_soft_body.update(configured_material)
    report["sloth_soft_body"] = {
        "model": "PBD.Elastic", "configuration": "scene.genesis.sloth_material",
        "parameters": sloth_soft_body.copy(),
        "mass_kg": sc["sloth"]["mass"],
        "calibrated_to_real_material": False,
    }
    materials = {
        "sloth": gs.materials.PBD.Elastic(
            rho=sloth_density,
            static_friction=sc["sloth"]["friction"], kinetic_friction=sc["sloth"]["friction"],
            **sloth_soft_body),
        "garment": gs.materials.PBD.Cloth(
            rho=garment_density, stretch_compliance=1e-7, bending_compliance=1e-5,
            **({"stretch_relaxation": cotton["stretch_relaxation"],
                "bending_relaxation": cotton["bending_relaxation"]} if cotton else {}),
            static_friction=sc["garment"]["friction"], kinetic_friction=sc["garment"]["friction"]),
    }
    deformables = {}
    with np.load(prepared / "physics_meshes.npz", allow_pickle=False) as archive:
        physics = {key: archive[key].copy() for key in archive.files}
    for name in ("sloth", "garment"):
        surface = (gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(image_path=str(paths["sloth_texture"])),
            vis_mode="visual", roughness=.9)
            if name == "sloth" else gs.surfaces.Default(color=(.12, .48, .26, 1),
                                                       vis_mode="visual", double_sided=True))
        with existing_pbd_mesh(name, physics[name + "_points"],
                               physics[name + ("_tetra" if name == "sloth" else "_faces")],
                               sc[name]["mass"]):
            deformables[name] = scene.add_entity(
                name=name, material=materials[name],
                morph=gs.morphs.Mesh(file=str(prepared / f"{name}.obj"),
                                     file_meshes_are_zup=True, decimate=False, convexify=False),
                surface=surface)
        sampled = deformables[name].mesh.trimesh
        actual = topology(sampled.vertices, sampled.faces, allow_nonmanifold=name == "sloth")
        before = report["geometry"][name]["surface" if name == "sloth" else "prepared"]
        require_same_topology(before, actual)
        report["geometry"][name]["genesis_surface"] = actual
        report["geometry"][name]["genesis_particles"] = deformables[name].n_particles
        if name == "sloth":
            report["geometry"][name]["genesis_tetrahedra"] = deformables[name].n_elems
        report["geometry"][name]["mass_kg"] = sc[name]["mass"]
        print(f"{name}: {deformables[name].n_particles} particles, surface {actual}", flush=True)

    cameras = {}
    for name, camera in sc["cameras"].items():
        cameras[name] = scene.add_camera(res=(args.width, args.height), GUI=False,
            pos=camera["position"], lookat=camera["target"], fov=camera["fovy"])
    cameras["wrist"] = scene.add_camera(res=(args.width, args.height), GUI=False, fov=80)
    wrist_offset = np.diag([1., -1., -1., 1.])
    wrist_offset[:3, 3] = (.06, 0, .02)
    cameras["wrist"].attach(robots["left"].get_link("hand"), wrist_offset)
    if args.task in ("lift-sloth", "dress-sloth"):
        cameras["lift"] = scene.add_camera(
            res=(args.width, args.height), GUI=False,
            pos=(.15, -.95, 1.25), lookat=(-.18, -.08, .97), fov=43)
    print("Building Genesis scene (first run compiles kernels)...", flush=True)
    scene.build()
    if cotton:
        report["cotton_material"] = apply_cotton_parameters(deformables["garment"], cotton)
        used, boundary = tetra_boundary(physics["sloth_points"], physics["sloth_tetra"])
        report["surface_contact"] = install_surface_contacts(deformables, used[boundary],
            sc["genesis"]["surface_contact"])
    report["fidelity_checks"] = {}
    for name, entity in deformables.items():
        actual_points = entity.get_particles_pos().detach().cpu().numpy()
        if not np.allclose(actual_points, physics[name + "_points"], rtol=0, atol=1e-7):
            raise RuntimeError(f"{name}: Genesis changed the initial particle coordinates")
        expected_elements = physics[name + ("_tetra" if name == "sloth" else "_faces")]
        actual_elements = simulation_connectivity(entity, name)
        if not np.array_equal(np.sort(actual_elements, axis=1), np.sort(expected_elements, axis=1)):
            raise RuntimeError(f"{name}: Genesis changed source element connectivity")
        actual_mass = float(entity.get_mass().detach().cpu().reshape(-1)[0])
        if not np.isclose(actual_mass, sc[name]["mass"], rtol=1e-5):
            raise RuntimeError(f"{name}: Genesis changed the configured mass")
        report["geometry"][name]["mass_kg"] = actual_mass
        report["fidelity_checks"][name] = {"initial_coordinates_match": True,
                                          "element_connectivity_matches": True,
                                          "mass_matches": True}
    home = np.array([*sc["home"], .04, .04], dtype=np.float32)
    joints = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]
    motors = {}
    for side, robot in robots.items():
        dofs = [robot.get_joint(name).dofs_idx_local[0] for name in joints]
        motors[side] = dofs
        robot.set_dofs_position(home, dofs_idx_local=dofs)
        robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]), dofs)
        robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]), dofs)
        robot.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                                  np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]), dofs)
        robot.control_dofs_position(home, dofs)
    report["genesis"] = {"version": version("genesis-world"), "torch": torch.__version__,
        "backend": args.backend, "solver": "PBD Elastic + PBD Cloth / legacy rigid coupling with scene-local Coulomb friction",
        "step_dt_s": dt, "substeps": args.substeps, "substep_dt_s": sc["timestep"],
        "particle_diameter_m": args.particle_size,
        "materials": {name: material.model_dump() for name, material in materials.items()},
        "robot_gravity_compensation":{side:robot.gravity_compensation for side,robot in robots.items()},
        "robot_dofs": {side: robot.n_dofs for side, robot in robots.items()}}
    return scene, robots, deformables, cameras


def positions_stats(deformables, table_height):
    result = {}
    for name, entity in deformables.items():
        p = entity.get_particles_pos().detach().cpu().numpy()
        v = entity.get_particles_vel().detach().cpu().numpy()
        if not np.isfinite(p).all() or not np.isfinite(v).all():
            raise RuntimeError(f"{name} has non-finite particle state")
        result[name] = {"bounds_m": [p.min(0).tolist(), p.max(0).tolist()],
                       "centroid_m": p.mean(0).tolist(),
                       "centroid_velocity_m_s": v.mean(0).tolist(),
                       "rms_speed_m_s": float(np.sqrt(np.mean(np.sum(v * v, axis=1)))),
                       "max_speed_m_s": float(np.linalg.norm(v, axis=1).max()),
                       "table_penetration_m": float(max(0, table_height - p[:, 2].min()))}
        if name == "sloth":
            tet = entity._elems
            volumes = np.linalg.det(p[tet[:, 1:]]-p[tet[:, :1]])/6
            collapsed = int(np.count_nonzero(volumes <= 0))
            result[name].update(volume_ratio=float(volumes.sum()/entity._elems_vol_rest.sum()),
                                inverted_or_collapsed_tetrahedra=collapsed)
            if collapsed:
                raise RuntimeError(f"Sloth has {collapsed} inverted or collapsed tetrahedra")
    return result


def render_views(cameras, run, label):
    for name, camera in cameras.items():
        if name == "wrist":
            camera.move_to_attach()
        rgb, *_ = camera.render()
        rgb = np.asarray(rgb)
        if rgb.std() < 1:
            raise RuntimeError(f"Camera {name} produced an empty image")
        Image.fromarray(rgb).save(run / f"{label}-{name}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, default=HERE.parent / "bunnylab")
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--headless", action="store_true", help="No interactive viewer")
    parser.add_argument("--prepare-only", action="store_true", help="Validate/export assets without starting Genesis")
    parser.add_argument("--rpc", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, default=400, help="Settling steps; viewer remains open afterwards")
    parser.add_argument("--task", choices=("settle", "lift-sloth", "dress-sloth"), default="settle")
    parser.add_argument("--continue-lift", type=Path, help="For dress-sloth, initialize from a verified lift run using the same materials/assets")
    parser.add_argument("--continue-supported", type=Path, help="For dress-sloth, restore a saved state-supported.npz before garment grasping")
    parser.add_argument("--continue-opened", type=Path, help="For dress-sloth, restore state-opened-hem.npz before inserting the sloth")
    parser.add_argument("--continue-dressing", type=Path, help="Restore a recorded dressing stage with its grasp metadata")
    parser.add_argument("--lift-height", type=float, default=.22, help="Dual-arm lift distance in metres")
    parser.add_argument("--hold-seconds", type=float, default=1.5, help="Final airborne hold duration")
    parser.add_argument("--record-fps", type=int, default=20, help="Video capture rate for lift-sloth")
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--particle-size", type=float, default=.004, help="PBD collision particle diameter in metres")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", type=Path, help="New output directory; default runs/<timestamp>-dressing-<pid>")
    args = parser.parse_args()
    if args.steps < 0 or args.substeps < 1 or args.particle_size <= 0 or min(args.width, args.height) < 32:
        parser.error("Invalid steps, substeps, particle size, or resolution")
    if not 0.1 <= args.lift_height <= .35 or args.hold_seconds < 1 or not 1 <= args.record_fps <= 60:
        parser.error("Lift height must be 0.1–0.35 m, hold >= 1 s, and record FPS 1–60")
    if args.continue_lift and args.task != "dress-sloth":
        parser.error("--continue-lift requires --task dress-sloth")
    if args.continue_supported and (args.task != "dress-sloth" or args.continue_lift):
        parser.error("--continue-supported requires dress-sloth and cannot be combined with --continue-lift")
    if args.continue_opened and (args.task != "dress-sloth" or args.continue_lift or args.continue_supported):
        parser.error("--continue-opened requires dress-sloth and cannot be combined with another continuation")
    if args.continue_dressing and (args.task != "dress-sloth" or args.continue_lift or args.continue_supported or args.continue_opened):
        parser.error("--continue-dressing requires dress-sloth and cannot be combined with another continuation")
    if args.rpc:
        from genesis_route import serve_scene
        serve_scene(args)
        return
    run = (args.output or HERE / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"-dressing-{os.getpid()}")).resolve()
    run.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("GS_CACHE_FILE_PATH", str(HERE / ".cache" / "genesis"))
    os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache" / "matplotlib"))
    report = {"status": "starting"}
    started = time.monotonic()
    try:
        print(f"Output: {run}", flush=True)
        config, paths, prepared, report = prepare_assets(args.source_project.resolve(), run)
        (run / "report.json").write_text(json.dumps(report, indent=2))
        if args.prepare_only:
            print("PASS: same source assets, source layout, and garment openings", flush=True)
            return
        scene, robots, deformables, cameras = build_scene(config, paths, prepared, report, args)
        if args.continue_lift:
            from sloth_lift import restore_lift_pose
            restore_lift_pose(robots, deformables, report, args.continue_lift.resolve())
        elif args.continue_supported:
            from sloth_lift import restore_dressing_pose
            restore_dressing_pose(robots, deformables, report, args.continue_supported.resolve())
        elif args.continue_opened:
            from sloth_lift import restore_dressing_pose
            restore_dressing_pose(robots, deformables, report, args.continue_opened.resolve())
        elif args.continue_dressing:
            from sloth_lift import restore_dressing_snapshot
            restore_dressing_snapshot(robots, deformables, report, args.continue_dressing.resolve())
        report["initial_state"] = positions_stats(deformables, config["scene"]["table_height"])
        render_views(cameras, run, "initial")
        if args.task == "lift-sloth":
            from sloth_lift import run_lift
            completed = run_lift(scene, robots, deformables, cameras, config, report, run, args)
        elif args.task == "dress-sloth":
            from sloth_lift import run_dressing
            completed = run_dressing(scene, robots, deformables, cameras, config, report, run, args)
        else:
            completed = 0
            report["settling_history"] = [{"simulation_seconds": 0., "objects": report["initial_state"]}]
            for step in range(args.steps):
                if not args.headless and not scene.viewer.is_alive():
                    break
                scene.step()
                completed = step + 1
                if completed % 50 == 0:
                    report["settling_history"].append({
                        "simulation_seconds": completed * report["genesis"]["step_dt_s"],
                        "objects": positions_stats(deformables, config["scene"]["table_height"])})
                    print(f"Step {completed}/{args.steps}, simulation {completed * report['genesis']['step_dt_s']:.3f} s", flush=True)
        report["final_state"] = positions_stats(deformables, config["scene"]["table_height"])
        if any(state["table_penetration_m"] > 1e-4 for state in report["final_state"].values()):
            raise RuntimeError("Settling validation failed: particles penetrated the tabletop")
        for side, robot in robots.items():
            qpos = robot.get_qpos().detach().cpu().numpy()
            if not np.isfinite(qpos).all():
                raise RuntimeError(f"{side} robot state is non-finite")
        render_views(cameras, run, "settled")
        np.savez(run / "final_particles.npz", **{
            name: entity.get_particles_pos().detach().cpu().numpy()
            for name, entity in deformables.items()})
        after_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
        if after_hashes != report["asset_sha256"]:
            raise RuntimeError("A source asset changed during the scene run")
        report["fidelity_checks"]["source_assets_unchanged"] = True
        status = {"settle": "scene_verified", "lift-sloth": "lift_verified_constraint",
                  "dress-sloth": "dressing_geometry_verified_constraint"}[args.task]
        report.update(status=status, completed_steps=completed,
                      simulation_seconds=completed * report["genesis"]["step_dt_s"],
                      wall_seconds=time.monotonic() - started)
        (run / "report.json").write_text(json.dumps(report, indent=2))
        print(f"PASS: {args.task}, {completed} steps, finite states, {len(cameras)} camera views. Report: {run / 'report.json'}", flush=True)
        if not args.headless:
            print("Viewer stays open at the final state. Close the window or press Ctrl+C to exit.", flush=True)
            while scene.viewer.is_alive():
                scene.viewer.update()
                time.sleep(1 / 30)
    except KeyboardInterrupt:
        print("Stopped by user.", flush=True)
    except Exception as exc:
        report.update(status="failed", error=str(exc), wall_seconds=time.monotonic() - started)
        (run / "report.json").write_text(json.dumps(report, indent=2))
        (run / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        if "genesis" in sys.modules:
            sys.modules["genesis"].destroy()


if __name__ == "__main__":
    main()
