"""Run a small Genesis simulation; optionally load Franka and render a frame."""
import argparse
import importlib.metadata
import logging
from pathlib import Path

import numpy as np
import torch
import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--franka", action="store_true")
parser.add_argument("--render", action="store_true")
args = parser.parse_args()

print("Genesis:", importlib.metadata.version("genesis-world"), flush=True)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
if args.backend == "cuda":
    assert torch.cuda.is_available(), "PyTorch cannot access the NVIDIA GPU"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

requested_backend = gs.cpu if args.backend == "cpu" else gs.cuda
gs.init(backend=requested_backend, seed=0, logging_level=logging.WARNING)
assert gs.backend == requested_backend, "Genesis selected a different backend"
scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
scene.add_entity(gs.morphs.Plane())
box = scene.add_entity(gs.morphs.Box(pos=(0.6, 0, 1), size=(0.1, 0.1, 0.1)))
robot = None
if args.franka:
    robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
camera = None
if args.render:
    camera = scene.add_camera(res=(320, 240), pos=(2.5, -2.5, 1.8), lookat=(0.3, 0, 0.4), GUI=False)

print("Building scene (first run compiles simulation kernels)...", flush=True)
scene.build()
for _ in range(100):
    scene.step()
position = box.get_pos().detach().cpu().reshape(-1)
assert torch.isfinite(position).all(), position
assert 0.02 < float(position[2]) < 0.2, position
print("Box settled:", position.tolist(), flush=True)
if robot is not None:
    assert torch.isfinite(robot.get_qpos()).all()
    print("Franka loaded, DOFs:", robot.n_dofs, flush=True)
if camera is not None:
    rgb, *_ = camera.render()
    assert rgb.shape == (240, 320, 3), rgb.shape
    assert np.asarray(rgb).std() > 1, "Rendered image appears empty"
    from PIL import Image
    output = Path(__file__).resolve().parent / "franka-preview.png"
    Image.fromarray(np.asarray(rgb)).save(output)
    print("Rendered frame:", output, flush=True)
print(f"PASS: {args.backend}, 100 simulation steps", flush=True)
gs.destroy()
