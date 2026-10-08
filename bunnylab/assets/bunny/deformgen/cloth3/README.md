# DeformGen cloth3 — MuJoCo adaptation

Source: the downloaded `Zili2002/DeformGen-SimAssets` snapshot, specifically
`cloth3/log/phystwin/cloth3/data/cloth3_0001/final_data.pkl`. The complete original
files and revision/hash manifest remain under `assets/third_party/DeformGen-SimAssets`.

This is a cloth patch, not a reconstructed garment with sleeves. The first
observed frame contains 4,742 coloured points. `scripts/prepare_cloth3.py` rotates
it 180 degrees about X as in the source config, scales it to 75%, samples 14 mm
planar voxels, and triangulates the surface, discarding edges longer than 30 mm.
The result is 343 vertices and 632 triangles, about 25.1 x 24.8 cm. The PNG
projects the observed colours. Both shape and texture are coarse approximations.

Regenerate from the downloaded dataset with:

```bash
.venv/bin/python scripts/prepare_cloth3.py
```

Runtime loads only NPZ (`allow_pickle=False`) and PNG. The 55 g MuJoCo 2D flex
uses edge-length constraints to resist stretch; it can bend and drape. Distributed
1.5 mm sphere colliders contact the table, Panda and sloth. Cloth self-collision
is disabled, and point colliders do not provide a continuous contact surface.
This is a MuJoCo adaptation, not the original learned PhysTwin simulation.
