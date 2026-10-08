# DeformGen sloth — MuJoCo adaptation

Sources:
- https://huggingface.co/datasets/Zili2002/DeformGen-SimAssets
- https://huggingface.co/shashuo0104/phystwin-toy
- https://github.com/Zili2002/DeformGen/blob/main/assets/sources.yaml

The dataset card refers sloth physics assets to the upstream PhysTwin repository.
Both complete snapshots are in `assets/third_party`, including their READMEs and
revision/file manifests. The DeformGen dataset snapshot contains 2,242 files
(221,392,913 bytes; see manifest for authoritative individual sizes).

`sloth.npz` derives from `surface_points` and `interior_points` in
`phystwin-toy/data/sloth_0001/final_data.pkl`. `sloth.png` projects first-frame
`object_colors` onto the surface. Regenerate with:

```bash
.venv/bin/python scripts/fetch_deformgen.py
.venv/bin/python scripts/prepare_sloth.py
```

The converter rotates (x,y,z) to (x,z,-y), scales to 0.65, samples 12 mm voxels,
and tetrahedralizes the remaining points. Long bridges and slivers are removed;
connectivity is checked. Result: 551 vertices and 2,549 tetrahedra, approximately
26.6 x 23.9 x 6.0 cm. Texture projection and the coarse surface are approximations.

MuJoCo uses a 0.18 kg volumetric elastic flex (Young's modulus 8,000 Pa,
Poisson ratio 0.3, Rayleigh damping 0.01 s). These are hand-tuned simulation
parameters, not the learned PhysTwin dynamics or a reproduction of its material
identification. Distributed 2 mm sphere contacts support the toy on the table;
self-collision is disabled. The existing point-attachment assisted grasp remains.
The original upstream `.pkl`/`.pth` files are retained; runtime loads only the
converted NPZ with `allow_pickle=False` and PNG.
