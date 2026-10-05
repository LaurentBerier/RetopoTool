# RetopoTool

Seam-preserving polygon reduction for textured **GLB** assets: props, environment pieces, scans
and characters. It also does a high-to-low **normal-map bake** and builds **LOD ladders**. It is pure
Python and needs no GPU, Blender or network access.

- **Optimize**: brings a dense mesh down to a budget. Examples are a 2M-triangle photogrammetry rock,
  an AI-generated prop or a sculpted wall section. The UV layout is kept exactly, so textures do not
  tear at seams or slide off the surface. The lost relief is put back as a normal map.
- **LOD ladder**: one file per rung. For a static mesh the rungs are 50% / 25% / 10% of the source.
  For a rigged character they are 100k / 60k / 30k / 15k triangles, and the skin stays intact on
  every rung. Each rung is re-baked from the full-resolution source.
- **Quality instruments**: point-to-surface fidelity, UV drift in texels, and a headless textured
  render.

It is a decimator plus attribute reconstruction, **not** a quad re-mesher. That is deliberate: the
source UVs, base colour, metallic-roughness and skin stay valid. [docs/algorithm.md](docs/algorithm.md)
explains every step and the failures that shaped it.

### What a prop gets that a naive decimator breaks

- **UV seams, hard edges and material borders are protected.** On a flat face a seam costs
  nothing to the simplifier, so a naive collapse runs across it and smears a decal over the
  wall. Here seam vertices only ever collapse along the seam.
- **Several materials on one mesh** are decimated as one surface, so no cracks open at material
  borders.
- **Hard edges** (split normals) stay hard, with or without a bake.
- **Shared materials and textures** are respected:
  - Meshes that share a material share one baked map.
  - A tiling normal map used by other assets is never overwritten.
  - A material that is also used by something left untouched is cloned before it is changed.
- **Tiling, trim-sheet or overlapping UVs** cannot hold a per-surface bake, so the bake is skipped
  for that material and the reason is reported. The mesh is still decimated, and it keeps its
  original normal map.
- **The bake checks itself.** Each material's bake is kept only if it measurably brings the
  shading closer to the source. A low-poly kit piece whose relief already lives in its normal map
  keeps that map.
- **Authored normals are kept.** Custom or weighted normals are the basis of the bake, not
  recomputed from geometry.
- **Codecs are matched.** A WebP pipeline gets WebP maps back, so file sizes stay in line.
- **Vertex layouts**: shared vertex arrays, interleaved buffers, extra UV sets, custom attributes
  and quantized positions are all carried through.

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

## Command line

```bash
retopo stats rock.glb
retopo optimize rock.glb rock_opt.glb --target-triangles 20000
retopo optimize crate.glb crate_opt.glb --ratio 0.3 --no-bake
retopo lod crate.glb lods/                               # lod1 50%, lod2 25%, lod3 10%
retopo lod crate.glb lods/ --tier near:40% --tier far:5%:512
retopo lod hero_rigged.glb lods/                         # skinned: 100k/60k/30k/15k
retopo optimize hero.glb hero_opt.glb --profile character

retopo measure fidelity --prop crate.glb crate_opt.glb   # p2s + shading error, one region
retopo measure fidelity --quality lods/crate_lod_lod3.glb # shape / topology / skin seams
retopo measure uv-drift crate.glb crate_opt.glb --render cmp.png
retopo measure render crate_opt.glb front.png
```

A tier is `name:BUDGET[:texture_size]`. `BUDGET` is either a triangle count or a percentage of the
source. Leave out the texture size to keep the source textures.

Results go to stdout as JSON and logs go to stderr (add `-v` for progress). `python -m retopotool`
works too.

## Python API

```python
from retopotool import mesh_stats, optimize_glb, build_lod_ladder, LodTier

mesh_stats("rock.glb")
# {'triangles': 2104220, 'vertices': ..., 'primitives': 1, 'meshes': 1, 'materials': 1,
#  'has_skin': False, 'dense': True, 'optimizable': True, 'recommended_ratio': 0.19, ...}

# Unrigged mesh -> budget + baked normal map. Never overwrites its input.
stats = optimize_glb("rock.glb", "rock_opt.glb", target_triangles=20_000)
stats = optimize_glb("crate.glb", "crate_opt.glb", ratio=0.3)
stats = optimize_glb("hero.glb", "hero_opt.glb", profile="character")   # head/hands first
print(stats["bake_skipped"])        # e.g. [{'material': 2, 'skipped': 'UVs outside 0-1 ...'}]

# Any mesh -> LOD ladder written to lods/<stem>_lod_<tier>.glb
ladder = build_lod_ladder("crate.glb", "lods/")                     # static: 50/25/10 %
ladder = build_lod_ladder("crate.glb", "lods/",
                          [LodTier("near", ratio=0.4), LodTier("far", ratio=0.05, texture_size=512)])
ladder = build_lod_ladder("hero_rigged.glb", "lods/")               # skinned: character tiers
for rung in ladder["levels"]:
    print(rung["name"], rung["triangles"], rung["size_bytes"], rung["warnings"])
```

`profile` decides how the triangle budget is spread:
- `"prop"`: uniform. This is the default for `optimize_glb`.
- `"character"`: the head and hands of a Y-up A/T-pose humanoid keep a larger share.
- `"auto"`: character for a skinned file, prop otherwise. This is the default for
  `build_lod_ladder`.

`head_boost` and `hand_boost` override the profile.

Lower-level entry points, if you want to drive the policy yourself:

| Function | What it does |
|---|---|
| `decimate_source_glb(src, out, target_ratio, *, bake, profile, head_boost, hand_boost)` | One unrigged decimation, keeping `target_ratio` of the triangles |
| `decimate_rigged_glb(src, out, target_triangles, texture_size=None, bake=True, bake_reference=None, profile="auto", ...)` | One LOD rung, with the skin preserved if there is one. `texture_size` downscales textures and sets the bake resolution. `bake_reference` is the mesh to bake from, matched primitive by primitive |
| `build_skin_proxy(P, F, target_tris)` | Geometry-only decimation that also returns which proxy vertex each original vertex collapsed into |
| `bake_normal_map(...)` / `bake_normals.bake_texels(...)` | The array-level normal bake (`retopotool.bake_normals`) |
| `retopotool.mesh_quality` | Pure-numpy topology, triangle-shape and skin-seam metrics |

Every decimation returns a stats dict:
- triangle and vertex counts before and after, and file sizes;
- `normal_map_baked`, `bake_resolution`, `bake` (per primitive) and `bake_skipped` (per material,
  with the reason);
- `head_boost` / `hand_boost` as applied, and `head_vertex_share`;
- a JSON-safe `quality` block: sliver fraction, min-angle p5, edge uniformity, boundary /
  non-manifold edges (welded per mesh, so a crack between materials shows up), winding, and the
  repair counters.

`build_lod_ladder` logs a warning (it never fails) when a rung's quality is outside the expected
band. It skips a rung that would not reduce anything.

## What it accepts, and what it refuses

The input is a self-contained GLB with triangle primitives, in glTF conventions. Any number of
meshes, primitives, materials and nodes is fine. Materials, textures, samplers, nodes, extensions
and the skeleton pass through. Strips, fans, lines and points are copied unchanged. The binary
buffer is repacked, so dropped geometry and orphaned accessors actually shrink the file.

It refuses with `ValueError` rather than writing a subtly broken file when it finds:
- animations;
- more than one skin, or `JOINTS_1` (8-influence skins);
- morph targets;
- Draco or meshopt compression, or GPU instancing (decompress first, e.g.
  `gltf-transform copy in.glb out.glb`);
- sparse accessors;
- multi-buffer GLBs;
- more than 5M vertices;
- an output path that does not end in `.glb`;
- for `optimize_glb` / `decimate_source_glb`, any skin at all (optimize before you rig).

It raises `NothingToDecimate` (a `ValueError`) when no triangle primitive has at least 64 faces.

With seam lock on (the default for props), a piece whose UV seams alone need more triangles than
the budget stops above the target instead of smearing its texture. Its stats then report
`seam_limited: true` next to `target_triangles`.

The normal-map bake is skipped, per material, when:
- the UVs leave the 0–1 square (tiling textures, trim sheets);
- more than 10% of the UV layout overlaps itself;
- the normal map samples `TEXCOORD_1` or uses `KHR_texture_transform`;
- the primitive has no `TEXCOORD_0` or no material;
- the baked result does not lower the shading error against the source (`check` in the stats has
  both numbers).

Decimation still happens, and `bake_skipped` names the reason.

## Tuning (environment variables)

Defaults are the measured optimum; most integrations never touch these. Most are read **at import
time**, so set them before importing `retopotool`. `RETOPO_OPT_BAKE_NORMALS` is read on each call.

| Variable | Default | Effect |
|---|---|---|
| `RETOPO_OPT_HEAD_BOOST` / `RETOPO_OPT_HAND_BOOST` | 3.0 / 2.0 | Importance warp for Optimize with `profile="character"` |
| `RETOPO_LOD_HEAD_BOOST` / `RETOPO_LOD_HAND_BOOST` | 2.0 / 1.6 | Importance warp for character LOD rungs |
| `RETOPO_UV_RESAMPLE` | 1 | Re-read UVs from the source surface at moved vertices (0 = verbatim copy) |
| `RETOPO_UV_RESAMPLE_ROUNDS` / `RETOPO_UV_ESCALATE_ROUNDS` / `RETOPO_UV_GATHER_MAX` | 12 / 6 / 64 | UV walk budgets |
| `RETOPO_LOD_SKIN_BLEND` | 1 | Blend skin over collapse clusters (0 = anchor row verbatim) |
| `RETOPO_LOD_SKIN_SIGMA` / `RETOPO_LOD_SKIN_ANCHOR_LOCK` | 1.0 / 0.9 | Blend falloff, and the contact-boundary guard |
| `RETOPO_LOD_QUALITY_PASS` | 1 | Edge-flip triangle-shape pass |
| `RETOPO_LOD_FLIP_CREASE_DEG` / `RETOPO_LOD_FLIP_GAIN_DEG` / `RETOPO_LOD_FLIP_ROUNDS` | 20 / 1.0 / 4 | Flip-pass guards |
| `RETOPO_OPT_BAKE_NORMALS` | 1 | Global kill-switch for the normal-map bake (the `bake=` argument is the per-call one) |
| `RETOPO_SEAM_LOCK` | prop: 1, character: 0 | Keep collapses from crossing UV seams, hard edges and material borders |
| `RETOPO_BAKE_CHECK` | 1 | Keep a baked map only if it improves shading (0 = always keep it) |

## Performance

Measured on one CPU:
- 1M → 400k triangles: ~20–25 s, including the bake.
- 3M → 400k triangles: ~35 s.
- One 30k LOD rung from a 400k rig: ~5 s.
- A 2k–19k-triangle kit prop (real game assets) to 25%: ~2–3 s, including the bake and its
  self-check.

Peak memory is about 3 GB for a 1M-triangle input.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The suite builds its meshes synthetically (`tests/_fixtures.py` for characters,
`tests/_prop_fixtures.py` for props) and needs no asset files. It pins the regressions documented in
[docs/algorithm.md](docs/algorithm.md). For characters:
- unit normalization, and a warp that cannot fold;
- no inverted faces, and UVs on the surface;
- a skin identical across UV seams, and the five silent rig breakers;
- the `.glb` extension trap.

For props:
- no cracks between materials, and hard edges kept;
- shared accessors and materials handled;
- tiling and overlapping UVs left unbaked;
- extra attributes carried through.

## License

MIT. See [LICENSE](LICENSE).
