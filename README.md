# RetopoTool

Seam-preserving polygon reduction for textured game characters in **GLB**, plus a high-to-low
**normal-map bake** and **skin-preserving LOD ladders**. It is pure Python and needs no GPU,
Blender or network access.

- **Optimize**: brings a dense mesh (for example a 1–3M-triangle AI-generated character) down to a
  workable budget (400k by default). The UV layout is kept exactly, so the texture does not tear at
  seams or slide off the surface, and the lost relief is put back as a normal map.
- **LOD ladder**: turns a rigged character into high / medium / low / minimum rungs (100k → 15k
  triangles). The skin stays intact on every rung, and each rung is re-baked from the
  full-resolution mesh. On a real character, 38 MB → 1.7 MB.
- **Quality instruments**: per-region fidelity, UV drift in texels, and a headless textured render.

It is a decimator plus attribute reconstruction, **not** a quad re-mesher. That is deliberate: the
source UVs, base colour, metallic-roughness and skin stay valid. [docs/algorithm.md](docs/algorithm.md)
explains every step and the failures that shaped it.

## Install

```bash
pip install git+https://github.com/LaurentBerier/RetopoTool.git
# or, from a clone:
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Tested on 3.10 and 3.13. Dependencies:

| Package | Version |
|---|---|
| numpy | — |
| scipy | — |
| pygltflib | — |
| Pillow | — |
| fast-simplification | ≥ 0.1.13, < 0.2 |
| libigl | == 2.6.2 (pinned) |

## Python API

```python
from retopotool import mesh_stats, optimize_glb, build_lod_ladder, LodTier

mesh_stats("hero.glb")
# {'triangles': 3061204, 'vertices': ..., 'has_skin': False, 'dense': True,
#  'optimizable': True, 'recommended_ratio': 0.131, ...}

# Unrigged mesh -> ~400k triangles + baked normal map. Never overwrites its input.
stats = optimize_glb("hero.glb", "hero_opt.glb")                  # auto ratio from the budget
stats = optimize_glb("hero.glb", "hero_200k.glb", target_triangles=200_000)
stats = optimize_glb("crate.glb", "crate_opt.glb", ratio=0.3,
                     head_boost=1.0, hand_boost=1.0)               # not a humanoid: uniform budget

# Rigged (or unrigged) mesh -> LOD ladder written to lods/hero_lod_<tier>.glb
ladder = build_lod_ladder("hero_rigged.glb", "lods/")             # default 4 tiers
ladder = build_lod_ladder("hero_rigged.glb", "lods/",
                          [LodTier("mobile", 20_000, 1024), LodTier("far", 5_000, 256)])
for rung in ladder["levels"]:
    print(rung["name"], rung["triangles"], rung["size_bytes"], rung["warnings"])
```

Lower-level entry points, if you want to drive the policy yourself:

| Function | What it does |
|---|---|
| `decimate_source_glb(src, out, target_ratio, *, bake, head_boost, hand_boost)` | One unrigged decimation, keeping `target_ratio` of the triangles |
| `decimate_rigged_glb(src, out, target_triangles, texture_size=None, bake=True, bake_reference=None, ...)` | One skin-preserving LOD rung. `texture_size` downscales textures and sets the bake resolution. `bake_reference` is the mesh to bake from |
| `build_skin_proxy(P, F, target_tris)` | Geometry-only decimation that also returns which proxy vertex each original vertex collapsed into. Useful for solving skin weights on a proxy and expanding them back |
| `bake_normal_map(...)` | The array-level normal bake (`retopotool.bake_normals`) |
| `retopotool.mesh_quality` | Pure-numpy topology, triangle-shape and skin-seam metrics |

Every decimation returns a stats dict:
- triangle and vertex counts before and after, and file sizes;
- `normal_map_baked` and `bake_resolution`;
- `head_vertex_share`;
- a JSON-safe `quality` block: sliver fraction, min-angle p5, edge uniformity, boundary /
  non-manifold edges, winding, and the repair counters.

`build_lod_ladder` logs a warning (it never fails) when a rung's quality is outside the expected
band.

## Command line

```bash
retopo stats hero.glb
retopo optimize hero.glb hero_opt.glb                    # auto ratio, 400k budget
retopo optimize hero.glb hero_opt.glb --target-triangles 200000 --no-bake
retopo lod hero_rigged.glb lods/                         # default tiers
retopo lod hero_rigged.glb lods/ --tier mobile:20000:1024 --tier far:5000:256

retopo measure fidelity hero.glb hero_opt.glb            # regional p2s + shading error
retopo measure fidelity --quality lods/hero_lod_low.glb  # shape / topology / skin seams
retopo measure uv-drift hero.glb hero_opt.glb --render cmp.png
retopo measure render hero_opt.glb front.png
```

Results go to stdout as JSON and logs go to stderr (add `-v` for progress). `python -m retopotool`
works too.

## What it accepts, and what it refuses

The input is a GLB with triangle primitives, in glTF conventions: Y-up, metres. Materials, textures,
nodes and the skeleton pass through. The binary buffer is repacked, so dropped geometry and orphaned
accessors actually shrink the file.

It refuses with `ValueError` rather than writing a subtly broken file when it finds:
- animations;
- more than one skin, or `JOINTS_1` (8-influence skins);
- morph targets;
- Draco or GPU instancing;
- sparse accessors;
- multi-buffer GLBs;
- more than 5M vertices;
- an output path that does not end in `.glb`;
- for `optimize_glb` / `decimate_source_glb`, any skin at all (optimize before you rig).

The head/hand **importance warp** assumes a humanoid in a rough A/T-pose. If it finds no head or
hands, that half of the warp is the exact identity. Pass `head_boost=1, hand_boost=1` to switch it
off explicitly.

## Tuning (environment variables)

Defaults are the measured optimum; most integrations never touch these. Most are read **at import
time**, so set them before importing `retopotool`. `RETOPO_OPT_BAKE_NORMALS` is read on each call.

| Variable | Default | Effect |
|---|---|---|
| `RETOPO_OPT_HEAD_BOOST` / `RETOPO_OPT_HAND_BOOST` | 3.0 / 2.0 | Importance warp for Optimize |
| `RETOPO_LOD_HEAD_BOOST` / `RETOPO_LOD_HAND_BOOST` | 2.0 / 1.6 | Importance warp for LOD rungs |
| `RETOPO_UV_RESAMPLE` | 1 | Re-read UVs from the source surface at moved vertices (0 = verbatim copy) |
| `RETOPO_UV_RESAMPLE_ROUNDS` / `RETOPO_UV_ESCALATE_ROUNDS` / `RETOPO_UV_GATHER_MAX` | 12 / 6 / 64 | UV walk budgets |
| `RETOPO_LOD_SKIN_BLEND` | 1 | Blend skin over collapse clusters (0 = anchor row verbatim) |
| `RETOPO_LOD_SKIN_SIGMA` / `RETOPO_LOD_SKIN_ANCHOR_LOCK` | 1.0 / 0.9 | Blend falloff, and the contact-boundary guard |
| `RETOPO_LOD_QUALITY_PASS` | 1 | Edge-flip triangle-shape pass |
| `RETOPO_LOD_FLIP_CREASE_DEG` / `RETOPO_LOD_FLIP_GAIN_DEG` / `RETOPO_LOD_FLIP_ROUNDS` | 20 / 1.0 / 4 | Flip-pass guards |
| `RETOPO_OPT_BAKE_NORMALS` | 1 | Global kill-switch for the normal-map bake (the `bake=` argument is the per-call one) |

## Performance

Measured on one CPU, from the real characters used in development:
- 1M → 400k triangles: ~20–25 s, including the bake.
- 3M → 400k triangles: ~35 s.
- One 30k LOD rung from a 400k rig: ~5 s.

Peak memory is about 3 GB for a 1M-triangle input.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The suite builds its meshes synthetically (`tests/_fixtures.py`) and needs no asset files. It pins
the regressions documented in [docs/algorithm.md](docs/algorithm.md): unit normalization, a warp that
cannot fold, no inverted faces, UVs on the surface, a skin identical across UV seams, the five silent
rig breakers, and the `.glb` extension trap.

## License

MIT. See [LICENSE](LICENSE).
