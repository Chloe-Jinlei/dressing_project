"""Franka viewer example, using the NVIDIA GPU by default."""
import argparse
import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--steps", type=int, default=1000)
args = parser.parse_args()
if args.steps < 1:
    parser.error("--steps must be positive")

gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu)
scene = gs.Scene(show_viewer=not args.headless)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
scene.build()
for _ in range(args.steps):
    scene.step()
gs.destroy()
