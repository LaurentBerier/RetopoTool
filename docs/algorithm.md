# How retopotool works — and why each step is the way it is

This document is the design record behind `retopotool.decimate` and `retopotool.bake_normals`.
Most of the steps below exist because a simpler version shipped, looked fine to every static check,
and was visibly broken on a real character. The measurements in §1–§11 are from real AI-generated
humanoids (1.8 m tall, glTF Y-up, metres; 400k–3M triangles; textures 2–4K). §13 covers what had to
change for props and environment pieces, which is what the tool is mostly used on.

The engine is **decimation + attribute reconstruction + normal-map bake**, not a quad re-mesh. It
keeps the source UV layout exactly, so base colour and metallic-roughness stay valid untouched.
Only the normal map is rewritten, and a re-unwrap is never needed.

---

## 1. Pipeline per primitive (`_decimate_primitive`)

1. **Weld vertices by position.** On generated meshes, about 40% of vertices are UV-seam splits:
   the same position carrying different UVs. Simplifying the welded mesh makes both sides of a seam
   collapse together, so no cracks open up.
2. **Importance warp** (`_importance_warp`). The head (above 0.84 of the height) and the hands
   (beyond 0.75 of the half-width) are magnified in a copy of the positions used **only** to steer
   the error metric (§3).
3. **Unit normalization.** The copy is rescaled so its bounding box spans `_SIMPLIFY_SPAN = 2000`
   units (§2).
4. **Quadric edge collapse**: `fast_simplification.simplify(target_count, agg=7,
   return_collapses=True)`, followed by `replay_simplification`. The replay gives the exact mapping
   from welded vertex to output vertex, plus the **quadric-optimal merged position** of every
   collapse cluster.
5. **Emit the merged position**, mapped back through the exact inverse of the warp and the rescale.
   This step is not negotiable (§4).
6. **Choose the attribute anchor**: the cluster member nearest the emitted position.
7. **Remap the original raw faces** through raw → weld → output and drop degenerates. Every
   surviving face descends from one original face, so each corner has an original ancestor vertex.
   Record each face's original normal *now*, before any flips.
8. **Edge-flip quality pass** (`_flip_round`) improves triangle shape without moving any vertex (§6).
9. **Solidity repairs** (§5): re-place vertices that invert a face, drop zero-volume flaps, then
   flip whatever is still inverted.
10. **Wedge re-split** (island-aware): an output vertex gets one attribute set per UV-island side.
    Each corner takes the raw split vertex of its anchor that lies on the same UV island as its
    ancestor corner. If the anchor has no split on that island (the collapse crossed a seam), the
    corner keeps the ancestor's own attributes. A plain nearest-UV pick there grabs an unrelated
    island and smears atlas swaths across faces: 5.3% of triangles crossed islands before this
    rule, 0.022% after.
11. **UV re-derivation** (`_resample_uv_on_source`, §7): the chart is chosen by exact copy, but the
    UV value is re-read from the source surface at the emitted position.
12. **Skin resolved per welded position** and blended over the collapse cluster (§8).
13. **Normal-map bake**, repack and write `.glb`. For rigged output, `_assert_skin_intact` then
    re-opens the written file.

Attributes (normal, tangent, colour) are copied **exactly** from one original vertex and are never
interpolated. Skin and UV are the two deliberate exceptions, and §7 and §8 explain why.

## 2. The simplifier was running in the wrong units

`fast_simplification` sweeps edges against an **absolute** error-threshold schedule
(~`1e-9 * (iter+3)^agg`), not in strict quadric-error order. A character in metres has ~2 mm edges
whose quadric errors (~1e-8) fall below the very first threshold. Almost the whole mesh qualifies on
sweep 1, and the collapse order degenerates to index order, which is close to random decimation.
Rescaling the copy to a 2000-unit span fixes it, and the same mesh at two scales now decimates
identically. That is pinned by `test_unit_normalization_*`.

`_SIMPLIFY_AGG = 7`: lower values never reach the target and stall at a fixed triangle count, no
matter how many stages are used.

## 3. Spend the budget where people look: the importance warp

Uniform quadric decimation gave the head 8% of the surviving vertices. The warp magnifies the head
(×3 for Optimize, ×2 for LODs) and the hands (×2 / ×1.6) **only in the copy used to steer the
error metric**.

It is a 1-D **density integrated along an axis** (`_axis_density_warp`): `u → ∫s`, with the two
transverse axes scaled by `s(u)`. Its Jacobian is triangular with diagonal `(s, s, s)`, so
`det = s³ > 0` everywhere and **the map provably cannot fold**. The obvious alternative, scaling a
region about its centroid with a smooth falloff, has an unbounded displacement gradient. It
**inverted 13.4% of the character's surface area** while still producing plausible-looking output,
because the simplifier then protects the folds.

If no head or hands are found, that half of the warp is the exact identity, so non-humanoid meshes
are unaffected. Passing `head_boost=1, hand_boost=1` turns the warp off.

Two measurement traps gave confidently wrong answers here:
- A surface vertex's 1-ring is coplanar, so a Jacobian fitted to it is rank-deficient and its
  determinant is noise.
- A face **normal** transforms by the inverse-transpose and rotates freely under shear, so a
  reversed face normal is not evidence of a fold.

Probe the determinant volumetrically instead.

Optimize at 400k, 3.0M-triangle source:

| | uniform | + unit fix & bake | + importance warp |
|---|---|---|---|
| head share of vertices | 8.0% | 13.8% | **22.7%** |
| head shading error p50 / p95 | 9.93° / 42.36° | 1.78° / 11.89° | **1.50° / 9.14°** |
| body shading error p50 / p95 | 3.22° / 30.27° | 0.51° / 7.96° | **0.53° / 8.60°** |

The LOD boosts are lower because the warp's cost grows as the budget gets scarcer. At 30k, 3.0/2.0
bought ~0.01 mm of head p95 for ~20% of body error, and made the head's *worst case* 2–4× worse.

## 4. The merged position is not negotiable

`fast_simplification` accepts or rejects each collapse with a normal-flip test evaluated **at the
optimal position**. Emitting a surviving original vertex instead (a median 4.5 mm away, up to
139 mm) invalidates every one of those decisions, and faces fold over. With backface culling, an
inverted face is invisible: that showed up as "holes" and "floating shards". The mesh was
watertight; it was turning inside out.

| target | inverted faces, survivor emitted | optimum emitted | p2s p95 |
|---|---|---|---|
| 100k | 1.75% | 0.03% | 1.62 → 0.47 mm |
| 30k | 3.28% | 0.15% | 4.99 → 1.53 mm |
| 15k | 4.07% | 0.33% | 10.05 → 3.73 mm |

Two properties make this safe:
- **A vertex that absorbed no neighbour is returned bit-identical**, so the mesh is not globally
  resampled.
- There is **one position per output welded vertex**, so every UV-split copy lands on the same
  point and the seam stays welded.

## 5. The solidity pass

`fast_simplification` applies no topology test, and its flip test runs in *warped* space. A large
triangle that is correctly oriented there can still invert once un-warped. Three repairs run in
order:

1. **Re-place vertices whose optimum inverts a face** (`_repair_inverted_faces`). The candidates are
   the vertex's own collapse-cluster members, nearest to the optimum first, so a repaired vertex
   always lands on the original surface.
2. **Delete zero-volume flaps** (`_drop_zero_volume_flaps`): opposite-wound triangle pairs on the
   same three vertices. A pair is deleted only where that cannot leave a boundary edge.
3. **Flip whatever is still inverted.** A sub-centimetre shading error beats a see-through hole.

Result: zero inverted and zero duplicate faces at every rung, and **no boundary edge is ever
introduced**. A closed mesh goes in and a closed mesh comes out. Triangle counts land slightly
*under* target, because flap removal runs after the simplifier has reached its budget.
`stats["quality"]["winding_flipped"]` is the honest count of the folds underneath.

## 6. Triangle shape: the edge-flip pass

Quadric collapse optimizes surface error, and a sliver has tiny error while shading terribly.
`_flip_round` flips an edge when doing so raises the minimum angle by at least
`RETOPO_LOD_FLIP_GAIN_DEG`. It never crosses a crease sharper than `RETOPO_LOD_FLIP_CREASE_DEG`, never
moves a vertex, and never creates a non-manifold edge. At 30k, slivers (<10°) dropped from 9.05% to
5.49%.

Measured and rejected:
- **Tangential relaxation** measured worse: slivers 9.05% → 11.5%, p2s doubled. Hard-surface
  creases leave only ~16% of vertices free to move.
- **Lowering `agg` or staging the collapse** never reaches the target.

Two traps:
- The flip must be **oriented against `f0`**. The edge table holds *sorted* endpoints, and using
  them blind reversed about half the flips.
- Original face normals must be **captured before the flip pass**, or the solidity pass mistakes
  good faces for inverted ones.

## 7. The texture slid off the surface: UV re-derivation

A collapse emits the cluster's optimal position, but the wedge re-split handed it the UV of one
original vertex, **verbatim**. The vertex is no longer where that UV was measured, so the texture
slides by the whole collapse distance. Quadric error is cheapest on flat plates, and flat plates are
where logos and lettering live, so a chest decal became unreadable while every geometric metric
stayed clean.

**Copy to choose the atlas chart; re-read the value from the source surface.**
`_resample_uv_on_source` runs a **greedy point-location walk** over the *raw* face graph, starting
from the ancestor vertex. A seam duplicates its vertices, so the walk physically cannot cross to the
other side of a seam. Its descent is monotone, so it terminates by itself.
`RETOPO_UV_RESAMPLE_ROUNDS` is a budget, not the stopping rule. Wedges that stall on a patch
boundary (high-valence poles) escalate to a two-ring neighbourhood.

- **Do not filter by `_uv_islands` instead of walking.** A chart that wraps (a cylinder cut along
  one seam) has both sides of the cut in one island, and the seam split collapses.
- **Walking beats widening.** A fixed two-ring patch costs 17× the candidates and still leaves a
  47-texel worst case.
- **Clamping is not projection.** `_closest_on_triangles` takes the best of the three edge
  projections.

On 1.06M → 399k triangles with a 2048 px atlas, vertices more than 1 texel out of place went from
11.3% to 0.9%. `RETOPO_UV_RESAMPLE=0` reproduces the verbatim-copy output byte for byte.

## 8. Skin: one row per position, blended over the cluster

Skin used to ride the per-corner attribute copy, so **each UV-split copy of one position took its
weights from a different source vertex**. At 30k, 60% of seam weld groups disagreed. Once posed,
that opened a 5 mm (p95) / 15 mm (p99) gap along every seam. Every static check passed, because the
split copies are duplicate vertices rather than neighbours across an edge.

Skin is now resolved **once per output welded vertex** and blended over the collapse cluster
(`_blend_skin_over_clusters`: area × proximity, top 4, renormalized). Every emitted joint comes from
a cluster member. The contact-boundary guard (`RETOPO_LOD_SKIN_ANCHOR_LOCK = 0.9`) keeps the
anchor's verbatim row where it is confident and blending would change the dominant bone, such as at
the crotch and armpits, where averaging `thigh_l` with `thigh_r` gives a vertex that follows neither.
At 0.7 the guard fired on 52% of vertices and measured worse than no guard at all.

Seam split after the fix: **0.0%** at every rung, with a posed gap of 0.00 mm.

### Five things that break a rig silently

Each of these produces a GLB that loads fine and then deforms into garbage. All are pinned by
tests and checked on the written file by `_assert_skin_intact`.

1. `skin.inverseBindMatrices` is an **accessor index**, and the repack rebuilds the accessor table,
   so it must be re-pointed.
2. `JOINTS_0` must not be rescaled like a normalized attribute: joint 66 must not become 0.259.
3. `JOINTS_n` must stay `UNSIGNED_BYTE`/`UNSIGNED_SHORT`. A float VEC4 there is invalid glTF.
4. `JOINTS_0`/`WEIGHTS_0` must be retired along with the other replaced accessors, or the primitive
   keeps an accessor at the old vertex count.
5. 8-influence rigs (`JOINTS_1`) are **refused**. Normalizing each set separately would make the
   weights sum to 2.

## 9. Normal-map bake (`bake_normals.py`)

Decimation removes the 2–5 mm relief (eyelids, lips, wrinkles, folds). It is put back as a
tangent-space normal map baked from the high-poly. Per material:

1. Rebuild the low-poly shading basis: smooth normals from the **welded** low-poly (they cross UV
   seams, but not the source's hard edges — see §13) and Lengyel tangents per split vertex (they do
   not). **Both are written onto the mesh**, because a normal map is only valid in the frame it was
   baked in.
2. Rasterize the low-poly into UV space at the bake resolution.
3. Cast a cage ray per texel along ±N and keep the nearest hit whose high-poly normal agrees. A
   plain nearest hit picks up the far side of a limb.
4. **Compose with the source normal map** rather than overwriting it. Generated meshes ship 2–4K of
   micro-detail that no geometric bake can reproduce.
5. Dilate 16 px past island borders and encode as **PNG** (JPEG chroma subsampling damages the X/Y
   channels).

Each LOD rung bakes from the **full-resolution source**, not from the rung above, so error does not
accumulate down the ladder. At 30k the bake cut body shading error from 10.27° to 3.42° with the
geometry unchanged. If the bake fails, the output degrades to plain decimation (`normal_map_baked:
false`) instead of failing the whole run.

Ray queries use libigl (`igl.AABB`), pinned to 2.6.2. trimesh's ray queries need the optional
`rtree` package.

## 10. File-level guards

- `pygltflib` picks GLB or JSON **from the file extension**, so a `.glb.tmp` target is silently
  written as JSON plus a `.bin` sidecar. Output paths must end in `.glb`, and `build_lod_ladder`
  writes `.{stem}_{uuid}.tmp.glb` before renaming it into place.
- The buffer is **repacked**, and only accessors something still points at are copied. Rig writers
  that append instead of repacking leave the pre-rig geometry in the file, about 25% of a 38 MB rig.
- The decimator refuses (with a ValueError) animations, more than one skin, `JOINTS_1`, morph
  targets, Draco / meshopt compression or GPU instancing, sparse accessors, multi-buffer files, and
  more than 5M vertices. Non-triangle primitives pass through unchanged. A file with no triangle
  primitive of at least 64 faces raises `NothingToDecimate` (a `ValueError`); the LOD ladder skips
  that rung instead.

## 11. LOD ladder defaults

Measured on a 1.83 m character rigged at 399k triangles / 41.5 MB (point-to-surface p95 against the
full-resolution rig):

| tier | tris | textures | head | body | hands | size |
|---|---|---|---|---|---|---|
| high | 100k | 2048 | 0.23 mm | 0.53 mm | 0.73 mm | 15.7 MB |
| medium | 60k | 1024 | 0.36 mm | 0.83 mm | 1.07 mm | 5.4 MB |
| low | 30k | 1024 | 0.72 mm | 1.79 mm | 2.22 mm | 3.9 MB |
| minimum | 15k | 512 | 1.77 mm | 4.58 mm | 4.32 mm | **1.7 MB** |

Those are the defaults for a **skinned** file. A static file gets `DEFAULT_PROP_LOD_TIERS`, budgets
relative to the source (50% / 25% / 10%) with the textures left as they are (§13).

## 12. Instruments

Point-to-surface error is a near-useless acceptance test on its own: a sliver sits right on the
surface and still shades like a crack. Judge changes by **region**, by **shading**, and by the
**render**:

- `retopo measure fidelity SRC OUT... [--quality] [--prop]`: per-region (head/hands/body, or one
  `all` region with `--prop`) p2s in mm, vertex share, and shading-normal error including the baked
  map. `--quality` adds triangle shape, topology and the seam-skin consistency check. Every triangle
  primitive is measured, placed by its node transform.
- `retopo measure uv-drift SRC OUT... [--render out.png]`: how far the texture slid, in texels.
- `retopo measure render GLB out.png`: a textured orthographic numpy render for before/after
  comparisons. Offline metrics have improved before while the render got worse.

## 13. Props and environment pieces

Everything above was built on one-primitive characters with one atlas. A prop is usually none of
that: several materials on one mesh, hard edges everywhere, materials and textures shared across
assets, tiling UVs. Each of these broke the character pipeline in its own way, each is now pinned by
`tests/test_props.py`:

| Layout | What went wrong | What happens now |
|---|---|---|
| One mesh, several materials (one primitive per material) | Each primitive was decimated on its own, so the two sides of a material border collapsed on different schedules and the border opened: 55 boundary edges on a two-material crate at 10%, 238 at the minimum LOD. | All triangle primitives of a mesh are decimated **jointly** (`_decimate_mesh_jointly`): the weld joins a material border exactly like a UV seam, and every output face goes back to its ancestor's primitive. 0 boundary edges. |
| Hard edges (split normals) | The bake recomputed every normal smoothed across all coincident vertices, so a crate shaded like a soap bar (corner normals leaning to 0.72). | Smoothing is restricted by the normals the decimator carried from the source (`HARD_EDGE_DEG = 35`). The flip pass and the wedge re-split stay inside one **raw-graph island**, which splits at hard edges too, with or without UVs. |
| Primitives sharing one vertex array | Retiring the replaced primitive's accessors cut the other primitive's data: `IndexError` or a cross-wired file. | Each primitive carries only the vertices its faces use, and an accessor is dropped only when nothing that is kept still points at it. |
| Tiling UVs / trim sheets (UV outside 0–1) | Baked anyway: the rasterizer clipped everything outside the unit square and the shared tiling normal map was overwritten. | That material is not baked (`bake_skipped` says why). The mesh keeps its carried normals and its original normal map. |
| Overlapping UVs (stacked or mirrored islands) | Last writer wins per texel. | Not baked when more than 10% of the layout is reused (`_uv_overlap_frac`). |
| A normal map on `TEXCOORD_1` or with `KHR_texture_transform` | Baked in TEXCOORD_0 space and plugged into a slot that samples something else. | Not baked. |
| Several meshes sharing one material | Each primitive's bake replaced the material's map in turn; the last one won. | One map **per material**: every primitive's covered texels are composited, then dilated once. |
| A material (or its normal image) also used by something not decimated | Its map was replaced under the untouched geometry. | The baked map overwrites the image only when nothing else can see it; otherwise the material is cloned for the decimated primitives and gets a new image. |
| A floor or wall panel (flat in Y) | The cage was 1% of the **Y** extent, so ~78% of rays missed. | The cage is 1% of the largest extent: 100% hit rate on the floor fixture. |
| Extra attributes (`TEXCOORD_2`, `_FEATURE_ID_0`, ...) | Left pointing into the old accessor table. | Every `TEXCOORD_n` is resampled like UV0; any other attribute is an exact per-vertex copy in its original component type. |
| Interleaved (strided) vertex buffers | Read element by element in Python. | Read as one strided view. |
| `KHR_mesh_quantization` | Integer positions read raw. | `normalized` integers are dequantized on read and written back as float. |

### Seams on flat faces: constraint fins

A UV seam, hard edge or material border on a FLAT face costs nothing to a position-only quadric,
so the collapse runs straight across it, and the faces on one side get painted with the other
side's texels. Characters hid this (their seams sit on curved skin). Kit pieces do not: on a real
cement entrance at 25%, the "STAY BACK" graffiti decal was smeared across the wall, with 638 texels
of UV drift at p95. The UV resample (§7) cannot help, because the damage is in which faces survive.

`fast_simplification` has no constraint hook, so each seam edge gets one extra **fin** triangle
standing on it along the surface normal (`_seam_fins`). Its plane is the textbook perpendicular
constraint plane. Its two new edges have a single face, which makes the seam's vertices **border**
vertices, and the simplifier only collapses a border vertex into another border vertex: along the
seam, never across it. The fins are dropped after the replay. Most fins collapse away with their
seam edges, which the face target would otherwise count, so the real face count is measured after
the replay and the simplification re-run once with a corrected target.

Measured on the entrance at 25%: UV drift p95 dropped from 638 to 0 texels at the requested 4.7k
triangles, and the render matches the source except for the thin cables' geometry. Seam lock is on
for props and off for characters (`RETOPO_SEAM_LOCK=0/1` overrides). When a piece is so
seam-dense that the seams alone exceed the budget, it stops above the target and the stats say so
(`seam_limited`).

### Should this material be baked at all?

Two findings on the real kit:

- **The low-poly normal basis is the CARRIED normals** (the source normals copied by the
  decimator), not normals recomputed from the coarse geometry. Kit pieces ship authored, weighted
  normals that differ from recomputed ones by 6° median and 39° p95. Recomputing them made the map
  re-encode all of that, and the median shading error rose from 3.8° to 8.6°. Without source
  normals, smooth normals that respect hard edges are computed (`HARD_EDGE_DEG = 35`).
- **A bake can lose to no bake.** It repairs what the decimation broke, but it re-encodes the
  source map at 8 bits. On a dense piece the repair wins by far: a barrel goes from 16.0° to 6.7°
  and a tank from 12.0° to 3.7°. On a door whose relief already lived in its normal map, it loses:
  15.0° baked against 14.8° unbaked. So every material is checked (`_bake_helps`): the shading
  error against the source is measured with and without the bake on up to 20k sampled source
  points, and the bake is kept only if it is at least 3% better. The statistic is a mean trimmed at
  60°. Double-sided kit geometry sends up to 20% of nearest-surface lookups to the wrong face
  (~140° for both variants); a plain mean drowns in that noise, and a median ignores the very
  patches the bake is for. `RETOPO_BAKE_CHECK=0` keeps every bake.

The baked map is encoded like the map it replaces. A lossy WebP source gives lossy WebP at q90
(1.0° p50 error; 112 KB against 604 KB as PNG), a lossless one gives lossless, anything else gives
PNG. Writing PNG over WebP doubled the size of a 450 KB prop.

The humanoid **importance warp is off by default** (`profile="prop"`): on a barrel it just magnifies
whatever sits in the top 16% of the bounding box. `profile="character"` turns it on;
`build_lod_ladder` uses `"auto"`, which means character for a skinned file and prop otherwise.

A vertex whose collapse cluster absorbed nothing is emitted at its **exact** source position. Read
back through the float32 simplify space, it was off by a rounding step, enough to break bit-exact
seams with geometry that was not decimated.

The bake resolution follows the material's own textures: the source normal map, else the base
colour, else 2048. A prop with 512 px textures gets a 512 px map.

