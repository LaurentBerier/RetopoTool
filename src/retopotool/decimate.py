"""Seam-preserving polygon reduction for textured (and optionally skinned) GLB meshes.

Two entry points: `decimate_source_glb` optimizes an UNRIGGED mesh that is too dense to work with
(e.g. a raw AI-generated character before rigging), and `decimate_rigged_glb` writes one rung of a
runtime LOD ladder for a RIGGED mesh with its skin intact. A naive decimator that transfers UVs by
nearest-neighbour tears the texture along every UV seam (~40% of verts on generated meshes are seam
splits), so it cannot be used for either.

This module instead simplifies on the POSITION-WELDED mesh (so the two sides of a UV seam collapse
in lockstep — no cracks) and rebuilds per-corner attributes exactly via fast_simplification's
collapse REPLAY:

  1. weld raw vertices by position -> welded mesh
  2. quadric edge-collapse on the welded mesh (`simplify(..., return_collapses=True)`), run in a
     UNIT-NORMALIZED and IMPORTANCE-WARPED copy of the positions so the collapses are actually
     error-ordered and the head/hands keep a AAA-style share of the budget (see _SIMPLIFY_SPAN)
  3. `replay_simplification` -> indice_mapping (welded vert -> output vert) AND the quadric-optimal
     merged position of each collapse cluster. That position is emitted, mapped back through the
     inverse of step 2's warp/rescale. It must be: the simplifier's normal-flip test ran at that
     position, so emitting a surviving original vertex instead inverts faces (see `_decimate_primitive`).
  4. remap the ORIGINAL raw faces through (raw -> weld -> output), drop degenerates. Every
     surviving face is an original face, so each corner has an original "ancestor" raw vertex.
  5. wedge re-split: an output vertex needs one attribute set per UV island side. Each corner
     picks, among the raw split-vertices of its ANCHOR welded vertex (the cluster member nearest
     the emitted position), the one ON THE SAME UV ISLAND as its ancestor corner (nearest UV among
     those) — island-interior corners agree (vertices stay shared), seam corners split per side.
     When the anchor vertex has NO split on the ancestor's island (the collapse crossed a seam),
     the corner keeps the ancestor's own attributes at the new position instead — a plain
     UV-nearest pick there grabs an unrelated island and smears a random swath of the atlas across
     the face (visible as texture speckles). Attributes (UV/normal/tangent/color/skin) are copied
     EXACTLY from one original vertex — never interpolated or NN-transferred.

Scope guard: `decimate_source_glb` raises ValueError on skinned / animated / morph-target GLBs
(optimize BEFORE rigging); `decimate_rigged_glb` accepts a single 4-influence skin. Textures,
materials and the node hierarchy are preserved untouched; the binary buffer is repacked so dropped
geometry actually shrinks the file. The why behind each step is in docs/algorithm.md.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from pygltflib import (GLTF2, Accessor, BufferView, Buffer, Image, Texture,
                       NormalMaterialTexture, FLOAT, UNSIGNED_BYTE, UNSIGNED_SHORT, UNSIGNED_INT)

from .gltf_io import _acc, _CT, MAX_TOTAL_VERTS, _resize_image

logger = logging.getLogger(__name__)

# Above this many triangles `mesh_stats` flags a mesh as `dense`. Raw AI-generated characters are
# routinely >400k and that IS the detail the user asked for; rigging/editing tools typically degrade
# past ~0.5M tris, so the flag is reserved for the truly pathological ones.
DENSE_TRIANGLE_THRESHOLD = 800_000
# What Optimize aims for: comfortable for rigging/preview while keeping silhouette detail. At 250k
# generated characters visibly lost face/hand/finger relief; 400k keeps the fine detail regions
# readable (400k tris ~ 200k verts). Textures are untouched here by design — decimation and texture
# quality are deliberately decoupled.
TARGET_TRIANGLES = 400_000

class NothingToDecimate(ValueError):
    """The file has no triangle primitive large enough to reduce (fewer than 64 faces)."""


_KMAX = 16  # max seam-split candidates considered per welded vertex (typical splits are 2-4);
            # verts with more splits fall back to ancestor attributes via the island check

# --- UV at a MOVED vertex -----------------------------------------------------------------------
# A collapse emits the cluster's quadric-optimal position, and the wedge re-split below hands that
# position the UV of one ORIGINAL vertex, verbatim. Exact copying is right for deciding WHICH ATLAS
# CHART a wedge belongs to, and wrong for the UV itself: the vertex is no longer where the copied
# UV was measured, so the texture slides by the whole collapse distance. It is worst exactly where
# the quadric is cheapest — a flat armour plate carrying a logo has near-zero geometric error, gets
# stripped hardest, and its decal warps while the silhouette is reproduced perfectly.
# Measured on the reported character (1,062,424 -> 399,470 tris, a chest emblem and "ZUCK 3000"
# lettering): re-deriving each wedge's UV from the SOURCE surface at the emitted position, inside
# the wedge's own chart, restores the text and the logo at the same triangle count. The budget was
# never the problem, so no amount of importance-warp tuning would have fixed it.
# Set RETOPO_UV_RESAMPLE=0 for the verbatim-copy path (byte-identical to before this pass existed).
_UV_RESAMPLE = os.environ.get("RETOPO_UV_RESAMPLE", "1") not in ("0", "false", "False")
# WHERE THE NEW UV IS READ FROM: a greedy point-location WALK over the source mesh, starting at the
# wedge's ancestor raw vertex. Each round tests the faces touching the current vertex, keeps the
# closest point found so far, and re-centres on the nearest corner of the winning face. Every step
# is an edge of the RAW face graph, and that graph IS the chart decomposition — a seam duplicates
# its vertices, so the walk physically cannot arrive on the far side of one. That is what lets a
# seam's two sides be re-parameterized independently, each inside its own chart.
# Do NOT substitute the `_uv_islands` label for the walk. An island is a connected component of the
# raw face graph, and on a chart that WRAPS (a cylinder cut open along one seam) both sides of the
# cut are the same component — filtering candidates by island lets the two sides project onto each
# other's faces, agree on one UV and destroy the split. `test_lod_uvs_still_split_at_the_seam`
# catches exactly that; the walk does not have the failure mode at all.
# The descent is MONOTONE (a wedge only steps when the round strictly improved its best distance),
# so the seed sequence cannot cycle and the pass terminates on its own — the round budget is a
# budget, not the thing that stops it. Measured on the reported character it converges in 5 rounds
# with nothing left moving, and on a flat 95%-reduction panel — the longest collapses this pipeline
# makes — it reaches the closed-form answer exactly, where a single fixed 2-ring patch leaves a
# 47-texel worst case. Widening the patch instead of walking was measured and is the wrong trade:
# a 2-ring costs 17x the candidates per round to save one round (41 s vs 22 s on the 1M-tri
# character, for 11 microns of off-surface residual).
# Incidence is stored as CSR and gathered at each chunk's OWN widest degree, not padded to a global
# cap. A cap is where this went wrong twice: truncating a vertex's faces can drop the very step the
# walk needed, and at a pole the walk cannot route around the loss (its step re-centres on the pole
# itself and stalls) — measured, a fixed cap left 2.1 texels of error on a 48-spoke fan. Padding to
# the global maximum instead would make one pathological vertex pay for every row, so the width is
# per chunk: an ordinary mesh gathers ~12 columns and a chunk containing a fan gathers as many as
# that fan needs. `_UV_GATHER_MAX` only bounds the temporaries, at a width no triangle mesh reaches.
_UV_GATHER_MAX = int(os.environ.get("RETOPO_UV_GATHER_MAX", "64"))
_UV_RESAMPLE_ROUNDS = int(os.environ.get("RETOPO_UV_RESAMPLE_ROUNDS", "12"))
# A one-ring walk can STALL: if the closest point of the seed's own faces lies on the patch's
# boundary, the true seat is one ring further out, but the step re-centres on a corner the walk has
# already been to and the monotone rule then stops it. That is exactly what a high-valence pole
# does — every face is close to the target, so the pole keeps winning. The boundary test is the
# textbook signal (a seated wedge's closest point is triangle-INTERIOR), and it selects a narrow
# enough set that the expensive two-ring neighbourhood can be spent on just those: 13% of wedges on
# the reported character, for the same off-surface residual as running two-ring on ALL of them at
# a third of the cost, and it is the difference between 2.1 texels and an exact answer on a 48-spoke
# fan. The escalation gathers ~13x the columns, so it runs at a proportionally smaller chunk to keep
# peak memory flat.
_UV_ESCALATE_ROUNDS = int(os.environ.get("RETOPO_UV_ESCALATE_ROUNDS", "6"))
_UV_RESAMPLE_CHUNK = 20_000

# --- simplifier calibration -------------------------------------------------------------------
# fast_simplification (Forstmann) does NOT collapse in strict quadric-error order: it sweeps edges
# against an ABSOLUTE threshold schedule (~1e-9 * (iter+3)^agg). A character in METRES has ~2mm
# edges, so its quadric errors (~1e-8) sit below the first threshold and essentially the whole mesh
# qualifies on sweep 1 — the collapse order degenerates to index order and the result is close to
# random decimation. Rescaling the mesh so its longest bbox axis spans _SIMPLIFY_SPAN units puts the
# error distribution inside the schedule's working range and the collapses become error-ordered.
# Measured on a 3.0M-tri character decimated to 400k tris (point-to-surface error vs the source):
#   metres  -> head p95 1.141 mm, body p95 0.858 mm
#   x1000   -> head p95 0.382 mm, body p95 0.280 mm
# Same simplifier, same target, same runtime — 3x the accuracy purely from the unit change.
_SIMPLIFY_SPAN = 2000.0
# Aggressiveness passed to fast_simplification. 7.0 is its default; lower values grow the schedule
# more slowly and MEASURED worse here (agg<=3 never reaches the target inside the 100-iteration cap
# and returns a 2.3M-tri mesh). Do not lower without re-measuring the reached triangle count.
_SIMPLIFY_AGG = 7.0

# --- skin resolution at a collapsed vertex ------------------------------------------------------
# Blend the collapse cluster's weights instead of copying one member's row. Copying is what the
# non-skin attributes do (and must do — an exact UV copy is what keeps a seam pointing at its own
# atlas region), but a skin row is a SAMPLE OF A CONTINUOUS FIELD, and copying one sample makes the
# field piecewise-constant. As edges grow that shows up as neighbouring vertices riding different
# bones: measured edge weight-L1 p95 0.142 on the 400k rig vs 0.51 at 30k, and under a test pose the
# worst single torn triangle grows 7.3 cm2 -> 25.9 cm2. Blending re-smooths the field.
# Set RETOPO_LOD_SKIN_BLEND=0 to fall back to the verbatim anchor row.
_SKIN_BLEND = os.environ.get("RETOPO_LOD_SKIN_BLEND", "1") not in ("0", "false", "False")
# A cluster member this far (relative to the cluster's own spread) contributes ~nothing. Without a
# falloff a single distant member of a long collapse chain pulls a vertex onto a bone it never
# touched.
_SKIN_BLEND_SIGMA = float(os.environ.get("RETOPO_LOD_SKIN_SIGMA", "1.0"))
# CONTACT-BOUNDARY GUARD. Across a contact seam (crotch, armpit) the two sides are geometrically
# adjacent but belong to DIFFERENT limbs, and averaging thigh_l with thigh_r produces a vertex that
# follows neither. When the anchor is this confident about its dominant bone and the blend would
# move it to a different one, the anchor wins.
# 0.9, not the intuitive 0.7: at 0.7 the guard fires on 52% of vertices and MEASURED WORSE than no
# guard at all (worst posed torn triangle 25.6 cm2 vs 14.3 cm2 at 30k) — it was overriding the blend
# on ordinary limb interiors, not just at contact boundaries. At 0.9 the result is identical to
# removing the guard entirely on this character, so it costs nothing and still covers the case it
# was written for.
_SKIN_BLEND_ANCHOR_LOCK = float(os.environ.get("RETOPO_LOD_SKIN_ANCHOR_LOCK", "0.9"))


def _blend_skin_over_clusters(J_w: np.ndarray, W_w: np.ndarray, mass: np.ndarray,
                              d_opt_w: np.ndarray, vmap: np.ndarray, cand_start: np.ndarray,
                              cand_order: np.ndarray, anchor: np.ndarray, n_out: int
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Area/proximity-weighted skin blend over each collapse cluster.

    `J_w`/`W_w` are per-WELDED-vertex skin rows, `mass` their incident-area share, `d_opt_w` each
    welded vertex's distance to the position its cluster actually emitted, and
    `cand_order`/`cand_start` the cluster CSR (`_o` / `_cand_start`, already sorted nearest-first).
    Returns (joints, weights) per OUTPUT vertex, 4 influences, rows summing to 1.

    Every emitted joint index comes from a cluster member, so the result can never reference a bone
    absent from the source — the property `_assert_skin_intact`'s range check relies on.
    """
    n_bones = int(J_w.max()) + 1 if J_w.size else 1
    counts = np.diff(cand_start)
    # Per-cluster spread, used as the falloff scale. A cluster that absorbed nobody has spread 0 and
    # its single member keeps weight 1 regardless.
    sums = np.bincount(vmap, weights=d_opt_w, minlength=n_out)
    spread = sums / np.maximum(counts, 1)
    scale = np.maximum(spread * _SKIN_BLEND_SIGMA, 1e-12)
    contrib = np.maximum(mass, 1e-12) * np.exp(-(d_opt_w / scale[vmap]) ** 2)

    out_j = np.zeros((n_out, 4), dtype=np.int64)
    out_w = np.zeros((n_out, 4), dtype=np.float64)
    CHUNK = max(1, int(4_000_000 // max(n_bones, 1)))
    for s in range(0, n_out, CHUNK):
        e = min(s + CHUNK, n_out)
        lo, hi = int(cand_start[s]), int(cand_start[e])
        mem = cand_order[lo:hi]
        if len(mem) == 0:
            continue
        owner = np.repeat(np.arange(e - s), counts[s:e])
        flat = np.zeros((e - s) * n_bones, dtype=np.float64)
        c = contrib[mem]
        for k in range(J_w.shape[1]):
            np.add.at(flat, owner * n_bones + J_w[mem, k], c * W_w[mem, k])
        acc = flat.reshape(e - s, n_bones)
        take = min(4, n_bones)
        idx = np.argpartition(-acc, take - 1, axis=1)[:, :take]
        val = np.take_along_axis(acc, idx, axis=1)
        srt = np.argsort(-val, axis=1)
        idx = np.take_along_axis(idx, srt, axis=1)
        val = np.take_along_axis(val, srt, axis=1)
        # A cluster with fewer than 4 distinct bones leaves argpartition free to name arbitrary
        # zero-weight columns — bones the cluster never touched. They are inert (weight 0) but they
        # would make "the emitted joints are a subset of the cluster's" false, which is the property
        # that keeps `_assert_skin_intact`'s range check meaningful. Point them at the dominant bone.
        val = np.maximum(val, 0.0)
        idx = np.where(val > 0, idx, idx[:, :1])
        out_j[s:e, :take] = idx
        out_w[s:e, :take] = val

    tot = out_w.sum(1, keepdims=True)
    dead = (tot <= 1e-12).ravel()
    out_w = np.where(dead[:, None], 0.0, out_w / np.maximum(tot, 1e-12))

    # Anchor wins where it is confident and the blend would change the dominant bone, and wherever
    # the blend produced nothing at all.
    aj, aw = J_w[anchor], W_w[anchor]
    aw = aw / np.maximum(aw.sum(1, keepdims=True), 1e-12)
    a_dom = aj[np.arange(len(aj)), aw.argmax(1)]
    a_conf = aw.max(1)
    b_dom = out_j[:, 0]
    keep_anchor = dead | ((a_conf > _SKIN_BLEND_ANCHOR_LOCK) & (b_dom != a_dom))
    out_j[keep_anchor] = aj[keep_anchor]
    out_w[keep_anchor] = aw[keep_anchor]
    return out_j, out_w

# --- importance-driven density allocation ------------------------------------------------------
# Uniform QEM spends the triangle budget by curvature alone, which on a full-body character means
# the face keeps ~12% of the vertices at 400k — the reported "face loses too much detail". A AAA
# retopo instead ALLOCATES density: head and hands get a disproportionate share.
# We do that by simplifying a WARPED copy of the mesh (the warp only steers the error metric —
# emitted positions are mapped back through its exact inverse). The warp must be a SMOOTH SPATIAL MAP:
# a per-vertex scale factor (e.g. curvature-weighted) tears the mesh apart in warp space and the
# simplifier then protects the tears instead of the features (measured: body p95 4.9mm, and the
# target triangle count is never reached). A step mask has the same problem at the region boundary,
# so both regions ramp in over a smoothstep band.
# Measured at 400k tris, mm scale: head p95 0.382 -> 0.195 mm and the head's share of surviving
# vertices 20.8% -> 39.6%, for body p95 0.280 -> 0.359 mm (the budget has to come from somewhere).
_HEAD_BOOST = float(os.environ.get("RETOPO_OPT_HEAD_BOOST", "3.0"))
_HAND_BOOST = float(os.environ.get("RETOPO_OPT_HAND_BOOST", "2.0"))
# Where the head starts, as a fraction of bbox height, and the smoothstep ramp width below it.
_HEAD_Y_FRAC = 0.84
_WARP_BAND_FRAC = 0.08
# A hand cluster is the mesh beyond this fraction of the half-width, below the head line (A-pose).
_HAND_X_FRAC = 0.75
# The LOD ladder uses a GENTLER boost than the 400k Optimize step. Measured on a real armored
# character at three targets, sweeping head/hand boost (p2s error vs the full-resolution rig):
#     target  boost        head p95 / max      torso p95     legs p95
#     100k    3.0/2.0      0.119 / 1.28        0.295         0.314
#     100k    2.0/1.6      0.131 / 0.30        0.266         0.291
#      30k    3.0/2.0      0.388 / 2.90        1.007         1.052
#      30k    2.0/1.6      0.394 / 0.96        0.796         0.854
#      15k    3.0/2.0      0.907 / 4.07        2.544         2.647
#      15k    2.0/1.6      0.855 / 1.92        1.908         1.982
# The body improves ~20% at every rung for a head p95 cost of ~0.01 mm, and the head's WORST-CASE
# error improves 2-4x — at 3.0 the warp is aggressive enough to create localized artifacts of its
# own, so it was past the optimum even for the region it was protecting. The 400k Optimize step
# keeps 3.0/2.0: budget is plentiful there and that rung is what the original face-detail
# complaint was about.
# Density-allocation profiles (see `_resolve_boosts`). Props are the default: the humanoid warp
# only means something on a humanoid.
PROFILES = ("auto", "prop", "character")
_LOD_HEAD_BOOST = float(os.environ.get("RETOPO_LOD_HEAD_BOOST", "2.0"))
_LOD_HAND_BOOST = float(os.environ.get("RETOPO_LOD_HAND_BOOST", "1.6"))

# --- post-collapse triangle-quality pass --------------------------------------------------------
# Quadric collapse optimises for surface ERROR, not triangle SHAPE, so it happily leaves long thin
# slivers: measured 3.2% of faces under a 10 degree minimum angle on the 400k source, rising to
# 9-12% down the ladder. A sliver sits right on the surface (its point-to-surface error is tiny) but
# interpolates its vertex normals across a near-degenerate footprint, which on a specular armored
# character reads as a bright/dark streak.
# Edge flips fix shape WITHOUT MOVING ANY VERTEX, so UVs, skin and the exact-copy invariant are all
# untouched and the point-to-surface error can only change by the retriangulation itself.
# WHAT WAS TRIED AND REJECTED: tangential relaxation (Laplacian + reprojection onto the source).
# It is the textbook remedy and it MEASURED WORSE here — slivers 9.05% -> 11.5%, p2s p95 0.98 ->
# 1.97 mm at 30k. The reason is visible in the mesh: 31% of this character's edges are hard-surface
# creases, so after freezing creases and corners only 16% of vertices have any freedom left, and
# reprojecting the rest onto the source surface pulls them straight back into the slivers. Do not
# re-attempt relaxation on hard-surface characters without solving the crease case first.
_QUALITY_PASS = os.environ.get("RETOPO_LOD_QUALITY_PASS", "1") not in ("0", "false", "False")
# Never flip an edge whose two faces disagree by more than this: that edge IS an armor plate
# boundary, and flipping across it rounds off the hard surface. Measured at 30k, of the 6,789
# flippable edges touching a sliver, 2,950 are blocked by exactly this guard.
_FLIP_CREASE_DEG = float(os.environ.get("RETOPO_LOD_FLIP_CREASE_DEG", "20.0"))
# Only flip when the minimum angle of the pair improves by at least this much, so the pass converges
# instead of oscillating between two equally bad configurations.
_FLIP_MIN_GAIN_DEG = float(os.environ.get("RETOPO_LOD_FLIP_GAIN_DEG", "1.0"))
_FLIP_ROUNDS = int(os.environ.get("RETOPO_LOD_FLIP_ROUNDS", "4"))


def _summarize_quality(density: List[Dict], plans: List[Dict]) -> Dict:
    """Roll the per-primitive telemetry plus the written geometry into one JSON-safe dict.

    The shape metrics are the ones that separate a good rung from a bad one: point-to-surface error
    stays sub-millimetre even when the mesh looks broken, because a SLIVER sits right on the surface
    and only shades badly. `edge_max_over_p50` is the density-uniformity number — 11x on the rungs
    that prompted this work.
    """
    def total(key):
        return int(sum(int(d.get(key, 0) or 0) for d in density))

    out = {
        "inverted_faces_repaired": total("inverted_faces_repaired"),
        "zero_volume_flaps_dropped": total("zero_volume_flaps_dropped"),
        "winding_flipped": total("winding_flipped"),
        "quality_flips": total("quality_flips"),
        "skin_keyed_on_position": all(d.get("skin_keyed_on_position", True) for d in density),
        "skin_blended": any(d.get("skin_blended") for d in density),
        # UV re-derivation (see _UV_RESAMPLE). `uv_offsurface_*` is how far, in metres, the walk's
        # answer landed from the emitted position — the residual it could not read exactly, and the
        # quality signal to watch. `uv_resample_unconverged` is NOT that signal: it counts only the
        # wedges still moving when the round budget ran out.
        "uv_resampled": any(d.get("uv_resampled") for d in density),
        "uv_resample_fallback": total("uv_resample_fallback"),
        "uv_resample_rounds": max([int(d.get("uv_resample_rounds", 0)) for d in density], default=0),
        "uv_resample_unconverged": total("uv_resample_unconverged"),
        "uv_escalated": total("uv_escalated"),
        "uv_offsurface_p95": round(max([float(d.get("uv_offsurface_p95", 0.0)) for d in density],
                                       default=0.0), 6),
        "uv_offsurface_max": round(max([float(d.get("uv_offsurface_max", 0.0)) for d in density],
                                       default=0.0), 6),
    }
    if not plans:
        return out
    try:
        from .mesh_quality import (topology_report, triangle_quality, weld_by_position)
        # Welded PER MESH: the primitives of one mesh are one surface (a crack between two of its
        # materials must show up as boundary edges), but two meshes are separate objects in their
        # own local spaces and welding them together would invent non-manifold edges.
        by_mesh: Dict = {}
        for pl in plans:
            by_mesh.setdefault(pl.get("key", (0, 0))[0], []).append(pl)
        Ps, Fs, off = [], [], 0
        for group in by_mesh.values():
            P = np.concatenate([np.asarray(pl["pos"], dtype=np.float64) for pl in group])
            o, F = 0, []
            for pl in group:
                F.append(np.asarray(pl["faces"], dtype=np.int64) + o)
                o += len(pl["pos"])
            Pw, Fw, _ = weld_by_position(P, np.concatenate(F))
            Ps.append(Pw)
            Fs.append(Fw + off)
            off += len(Pw)
        Pw, Fw = np.concatenate(Ps), np.concatenate(Fs)
        q = triangle_quality(Pw, Fw)
        t = topology_report(Pw, Fw, welded=True)
        out.update({k: round(float(q[k]), 4) for k in
                    ("sliver_frac", "min_angle_p5", "edge_p50_mm", "edge_p99_mm",
                     "edge_max_mm", "edge_max_over_p50")})
        out.update({k: int(t[k]) for k in
                    ("boundary_edges", "nonmanifold_edges", "duplicate_faces",
                     "winding_inconsistent")})
    except Exception:
        # Telemetry must never fail a build that otherwise succeeded.
        logger.warning("retopotool: quality summary failed", exc_info=True)
    return out


def _tri_min_angle(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Smallest interior angle of each triangle, in degrees."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    e0 = np.linalg.norm(b - a, axis=1)
    e1 = np.linalg.norm(c - b, axis=1)
    e2 = np.linalg.norm(a - c, axis=1)

    def ang(opp, x, y):
        return np.degrees(np.arccos(np.clip((x * x + y * y - opp * opp) /
                                            (2 * np.maximum(x * y, 1e-15)), -1.0, 1.0)))

    return np.minimum(np.minimum(ang(e0, e1, e2), ang(e1, e2, e0)), ang(e2, e0, e1))


def _flip_round(pos: np.ndarray, Fo: np.ndarray, Fraw: np.ndarray, anc_n: Optional[np.ndarray],
                isl: Optional[np.ndarray], crease_cos: float, min_gain: float) -> int:
    """One greedy round of quality-improving edge flips. Mutates `Fo`/`Fraw` in place, returns the
    number applied. Guards are evaluated vectorised over every manifold interior edge; only the
    small surviving candidate set goes through a Python loop, so this stays milliseconds even at
    100k faces."""
    N, _ = _face_normals(pos, Fo)
    E = np.concatenate([Fo[:, [0, 1]], Fo[:, [1, 2]], Fo[:, [2, 0]]], axis=0)
    Es = np.sort(E, axis=1)
    fid = np.tile(np.arange(len(Fo)), 3)
    nv = int(Fo.max()) + 2
    code = Es[:, 0].astype(np.int64) * nv + Es[:, 1]
    order = np.argsort(code, kind="stable")
    cs, fs, es = code[order], fid[order], Es[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    runs = np.diff(np.r_[starts, len(cs)])
    two = starts[runs == 2]
    if len(two) == 0:
        return 0
    f0, f1 = fs[two], fs[two + 1]
    p, q = es[two, 0].copy(), es[two, 1].copy()
    # ORIENT THE SHARED EDGE WITH f0. `es` holds the SORTED endpoints, which throws away direction,
    # but the flip formula below is only orientation-preserving when p->q is a directed edge OF f0
    # (f0 = (p,q,r), f1 = (q,p,s) => (r,s,q) and (s,r,p) reuse each original directed edge exactly
    # once). Using the sorted pair blind reverses the new pair on roughly half the edges, which
    # leaves neighbouring faces disagreeing about which side is out — 535 inconsistent edges on a
    # real 30k rung, i.e. a backface-culled hole for every single-sided material.
    F0 = Fo[f0]
    has_pq = (((F0[:, 0] == p) & (F0[:, 1] == q)) |
              ((F0[:, 1] == p) & (F0[:, 2] == q)) |
              ((F0[:, 2] == p) & (F0[:, 0] == q)))
    p, q = np.where(has_pq, p, q), np.where(has_pq, q, p)
    # opposite corner of each face (vertices are distinct, so a sum difference identifies it)
    r = Fo[f0].sum(1) - p - q
    sv = Fo[f1].sum(1) - p - q

    ok = (r != sv)
    ok &= (np.einsum("ij,ij->i", N[f0], N[f1]) > crease_cos)          # never cross a plate crease
    newcode = np.minimum(r, sv).astype(np.int64) * nv + np.maximum(r, sv)
    ok &= ~np.isin(newcode, cs)                                       # would duplicate an edge
    if isl is not None:
        # all four corners must share a UV island, or the wedge re-split downstream would have to
        # bridge two unrelated atlas regions across the new edge
        i0, i1 = isl[Fraw[f0]], isl[Fraw[f1]]
        ok &= (i0.max(1) == i0.min(1)) & (i1.max(1) == i1.min(1)) & (i0[:, 0] == i1[:, 0])
    if not ok.any():
        return 0
    idx = np.flatnonzero(ok)
    T0 = np.stack([r[idx], sv[idx], q[idx]], axis=1)
    T1 = np.stack([sv[idx], r[idx], p[idx]], axis=1)
    old = np.minimum(_tri_min_angle(pos, Fo[f0[idx]]), _tri_min_angle(pos, Fo[f1[idx]]))
    new = np.minimum(_tri_min_angle(pos, T0), _tri_min_angle(pos, T1))
    n0, a0 = _face_normals(pos, T0)
    n1, a1 = _face_normals(pos, T1)
    good = (new > old + min_gain) & (a0 > 1e-14) & (a1 > 1e-14)
    good &= (np.einsum("ij,ij->i", n0, N[f0[idx]]) > crease_cos)      # no fold, no inversion
    good &= (np.einsum("ij,ij->i", n1, N[f0[idx]]) > crease_cos)
    idx = idx[good]
    if len(idx) == 0:
        return 0

    # Greedy: best gain first, one flip per face per round.
    gain = (new - old)[good]
    seq = idx[np.argsort(-gain)]
    used = np.zeros(len(Fo), dtype=bool)
    # `cs` is the edge table as it was at the START of the round, so the vectorised duplicate check
    # above cannot see edges introduced by flips accepted EARLIER in this same loop. Two candidates
    # that share no face can still share both opposite corners and therefore propose the identical
    # new diagonal; accepting both puts four faces on one edge. Track what we have added.
    added = set()
    applied = 0
    for e in seq:
        a, b = int(f0[e]), int(f1[e])
        if used[a] or used[b]:
            continue
        pp, qq, rr, ss = int(p[e]), int(q[e]), int(r[e]), int(sv[e])
        key = (rr, ss) if rr < ss else (ss, rr)
        if key in added:
            continue
        added.add(key)
        # Raw ancestor of each welded corner, taken from the face that corner already belonged to,
        # so `Fraw` stays a faithful ancestor map and the wedge re-split is unaffected.
        anc = {}
        for f in (a, b):
            for wv, rv in zip(Fo[f], Fraw[f]):
                anc.setdefault(int(wv), int(rv))
        Fo[a] = (rr, ss, qq)
        Fraw[a] = (anc[rr], anc[ss], anc[qq])
        Fo[b] = (ss, rr, pp)
        Fraw[b] = (anc[ss], anc[rr], anc[pp])
        if anc_n is not None:
            # Both new faces span the same two originals, which the crease guard has already proved
            # are within `crease_cos` of each other, so their mean is a faithful reference
            # orientation for the pair. Carrying it is what keeps the solidity pass meaningful on
            # flipped faces instead of having to skip them.
            m = anc_n[a] + anc_n[b]
            ln = float(np.linalg.norm(m))
            if ln > 1e-12:
                anc_n[a] = anc_n[b] = m / ln
        used[a] = used[b] = True
        applied += 1
    return applied


def mesh_stats(source_glb_path: str) -> Dict:
    """Cheap density analysis (accessor counts only — no buffer decode of vertex data)."""
    if not os.path.exists(source_glb_path):
        raise FileNotFoundError(source_glb_path)
    g = GLTF2().load(source_glb_path)
    tris = 0
    verts = 0
    prims = 0
    for mesh in (g.meshes or []):
        for prim in mesh.primitives:
            if prim.attributes.POSITION is None:
                continue
            prims += 1
            n = g.accessors[prim.attributes.POSITION].count
            verts += n
            if (prim.mode if prim.mode is not None else 4) == 4:   # lines/points/strips: no tris
                tris += (g.accessors[prim.indices].count // 3) if prim.indices is not None else n // 3
    has_skin = bool(g.skins)
    has_animation = bool(g.animations)
    optimizable = tris > 0 and not has_skin and not has_animation
    recommended_ratio = min(1.0, TARGET_TRIANGLES / tris) if tris else 1.0
    return {
        "triangles": int(tris),
        "vertices": int(verts),
        "primitives": int(prims),
        "meshes": len(g.meshes or []),
        "materials": len(g.materials or []),
        "file_size": os.path.getsize(source_glb_path),
        "has_skin": has_skin,
        "has_animation": has_animation,
        "dense": tris > DENSE_TRIANGLE_THRESHOLD,
        "optimizable": optimizable,
        "recommended_ratio": round(float(recommended_ratio), 3),
    }


def _uv_islands(n_raw: int, F: np.ndarray) -> np.ndarray:
    """UV-island label per raw vertex: connected components of the raw face graph. Seam edges are
    duplicated per side in the raw mesh, so faces only ever connect vertices within one island."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    adj = sparse.coo_matrix(
        (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(n_raw, n_raw)
    )
    _, labels = connected_components(adj, directed=False)
    return labels.astype(np.int64)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Hermite ramp, 0 below 0 and 1 above 1 — C1 at both ends, so a region boundary built from it
    leaves no density discontinuity for the simplifier to protect."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _axis_density_warp(P: np.ndarray, axis: int, density: np.ndarray, grid: np.ndarray,
                       centre: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply a 1-D density field along one axis as a PROVABLY fold-free warp.

    Given a density `s(u) >= 1` sampled on `grid` along `axis`, the map is

        u  ->  U(u) = integral of s      (strictly increasing, because s > 0)
        v  ->  v * s(u)   for the two other axes

    whose Jacobian is triangular with diagonal (s, s, s), so its determinant is s**3 > 0
    everywhere: the map is a diffeomorphism and no region can turn inside out, whatever shape the
    density has. The obvious alternative — scale a region about its centroid with a smooth falloff —
    has no such guarantee: its displacement gradient `(P - centre) . grad(w)` is unbounded, and the
    map inverts wherever it exceeds 1.

    `centre` fixes the transverse scaling's origin. It does not affect the determinant, but it does
    control SHEAR: the off-diagonal term is `(v - centre) * s'(u)`, so anchoring it on the region
    being magnified keeps `v - centre` small there instead of paying the model's whole width.
    """
    u = P[:, axis]
    integral = np.concatenate([[0.0], np.cumsum(np.diff(grid) * 0.5 * (density[1:] + density[:-1]))])
    Q = P.copy()
    Q[:, axis] = np.interp(u, grid, integral) + grid[0]
    s_at = np.interp(u, grid, density)
    c = np.zeros(3) if centre is None else np.asarray(centre, dtype=np.float64)
    for other in range(3):
        if other != axis:
            Q[:, other] = c[other] + (P[:, other] - c[other]) * s_at
    return Q


def _invert_axis_density_warp(Q: np.ndarray, axis: int, density: np.ndarray, grid: np.ndarray,
                              centre: Optional[np.ndarray] = None) -> np.ndarray:
    """Exact inverse of `_axis_density_warp` with the same arguments.

    The axis map is the integral of a strictly positive density, so it is strictly increasing and
    `np.interp` inverts it by swapping its x/y tables. The transverse scale is then read at the
    RECOVERED coordinate — reading it at the warped one would be a different (wrong) map.
    """
    integral = np.concatenate([[0.0], np.cumsum(np.diff(grid) * 0.5 * (density[1:] + density[:-1]))])
    u = np.interp(Q[:, axis] - grid[0], integral, grid)
    s_at = np.interp(u, grid, density)
    P = Q.copy()
    P[:, axis] = u
    c = np.zeros(3) if centre is None else np.asarray(centre, dtype=np.float64)
    for other in range(3):
        if other != axis:
            P[:, other] = c[other] + (Q[:, other] - c[other]) / s_at
    return P


def _inverse_importance_warp(Q: np.ndarray, steps: List[Dict]) -> np.ndarray:
    """Map positions from simplify space back to real space.

    `steps` is what `_importance_warp` returned — the arguments of each warp it applied, in order.
    Undoing them in REVERSE order is the inverse of their composition. Round trip measured at
    1.7e-12 mm on a real character, so a position that comes back through here is exact for every
    purpose downstream (the bake, the skin, the file).
    """
    P = np.asarray(Q, dtype=np.float64)
    for step in reversed(steps):
        P = _invert_axis_density_warp(P, step["axis"], step["density"], step["grid"],
                                      centre=step.get("centre"))
    return P


def _importance_warp(Pw: np.ndarray, head_boost: float = None, hand_boost: float = None
                     ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Positions to SIMPLIFY in, with the head and hands locally magnified.

    Magnifying a region multiplies its quadric errors by the same factor, so the simplifier
    collapses it later and it keeps a larger share of the triangle budget — the standard way to
    steer QEM without touching the simplifier itself.

    The magnification is expressed as a 1-D DENSITY along an axis and integrated (see
    `_axis_density_warp`), which makes the warp orientation-preserving by construction. The head is
    a density ramp along Y; the hands are ramps at both ends of X (an A-pose spans its arms along
    X). Composing two fold-free maps is fold-free.

    Returns (warped positions, head mask, steps). The mask is only used for telemetry. `steps`
    records the arguments of each warp applied, in order, so `_inverse_importance_warp` can bring a
    position computed in simplify space back to real space — the caller MUST NOT rebuild these
    parameters itself: the hand ramp is derived from the already-head-warped coordinates, and a
    second derivation drifts out of sync with this one.

    The character is assumed Y-up (glTF convention) in a rough A/T-pose, which is the expected
    input for the humanoid boosts. Where a region cannot be identified the density
    stays flat at 1.0 and that half of the warp is exactly the identity, rather than a guess.
    """
    head_boost = _HEAD_BOOST if head_boost is None else float(head_boost)
    hand_boost = _HAND_BOOST if hand_boost is None else float(hand_boost)
    lo, hi = Pw.min(0), Pw.max(0)
    height = float(hi[1] - lo[1])
    head = np.zeros(len(Pw), dtype=bool)
    steps: List[Dict] = []
    if height <= 0 or len(Pw) < 64:
        return Pw, head, steps

    Q = Pw.astype(np.float64, copy=True)
    y_head = lo[1] + _HEAD_Y_FRAC * height
    head = Pw[:, 1] > y_head
    band = max(_WARP_BAND_FRAC * height, 1e-9)

    if head_boost > 1.0 and head.sum() >= 16:
        gy = np.linspace(lo[1], hi[1], 512)
        s = 1.0 + (head_boost - 1.0) * _smoothstep((gy - (y_head - band)) / band)
        steps.append({"axis": 1, "density": s, "grid": gy, "centre": Pw[head].mean(0)})
        Q = _axis_density_warp(Q, 1, s, gy, centre=steps[-1]["centre"])

    if hand_boost > 1.0:
        x_abs = np.abs(Pw[:, 0])
        x_max = float(x_abs.max())
        # A hand only exists if the extremity is a distinct, small cluster — on a bust or a prop
        # the |x| extremes are just the body's own width and boosting them means nothing.
        hands = (~head) & (x_abs > _HAND_X_FRAC * x_max)
        if x_max > 0 and 16 <= hands.sum() <= 0.25 * len(Pw):
            # Place the ramp at the hand cluster's own INNER edge (in the current, possibly
            # head-warped, coordinates) and give it the cluster's own width. Deriving the threshold
            # from a fraction of the model's half-width instead puts the ramp outboard of most of
            # the hand: measured, that magnified the hand by 1.08x where 2.0x was asked for.
            hx = np.abs(Q[hands, 0])
            inner = float(hx.min())
            edge = max(float(np.percentile(hx, 90)) - inner, band) * 0.5
            gx = np.linspace(Q[:, 0].min(), Q[:, 0].max(), 512)
            s = 1.0 + (hand_boost - 1.0) * _smoothstep((np.abs(gx) - inner) / max(edge, 1e-9))
            steps.append({"axis": 0, "density": s, "grid": gx, "centre": None})
            Q = _axis_density_warp(Q, 0, s, gx)

    return Q, head, steps


def _weld(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Group raw vertices by (tolerance-quantized) position. Returns (weld_ids, n_weld)."""
    span = float(np.linalg.norm(P.max(0) - P.min(0)))
    tol = max(span, 1e-6) * 1e-6
    Pq = np.round(P.astype(np.float64) / tol).astype(np.int64)
    _, first, weld = np.unique(Pq, axis=0, return_index=True, return_inverse=True)
    return weld.astype(np.int64), len(first)


def build_skin_proxy(P: np.ndarray, F: np.ndarray, target_tris: int
                     ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Geometry-only decimation proxy for skin solving (no UV/attr replay — the proxy is internal).

    Welds by position, runs the same quadric edge-collapse as the delivery decimator, and returns
    (proxy_positions f64, proxy_faces i64, ancestor_map i64) where ancestor_map[raw_vert] is the
    proxy vertex whose collapse chain absorbed it. That map is an EXACT topological correspondence:
    transferring skin weights through it cannot land on the wrong side of a contact boundary
    (armpit/crotch) the way closest-point sampling between the two resolutions could.
    Returns None when the mesh is already at/under target or the simplifier cannot run.
    """
    import fast_simplification as _fs

    n_raw = len(P)
    if n_raw < 64 or len(F) <= target_tris:
        return None
    weld, n_weld = _weld(P)
    order = np.argsort(weld, kind="stable")
    starts = np.searchsorted(weld[order], np.arange(n_weld))
    Pw = P[order[starts]].astype(np.float64)
    Fw = weld[F]
    nz = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])
    Fw_clean = Fw[nz].astype(np.int32)
    if len(Fw_clean) < 64 or len(Fw_clean) <= target_tris:
        return None
    Pw32 = Pw.astype(np.float32)              # replay requires float32 points
    reduction = 1.0 - target_tris / float(len(Fw_clean))
    out = _fs.simplify(Pw32, Fw_clean,
                       target_reduction=float(np.clip(reduction, 0.05, 0.98)),
                       return_collapses=True)
    _, _, collapses = out
    if collapses is None or len(collapses) == 0:
        return None
    pts_r, faces_r, vmap = _fs.replay_simplification(Pw32, Fw_clean, collapses)
    vmap = np.asarray(vmap, dtype=np.int64)   # welded vert -> proxy vert
    # Validate the replay bookkeeping before anyone binds weights through this map (the same
    # guard `_decimate_primitive` applies). A silently wrong ancestor map would not crash — it
    # would hand every vertex the weights of an unrelated part of the body.
    if len(vmap) != n_weld or vmap.min() < 0 or vmap.max() >= len(pts_r):
        logger.warning("build_skin_proxy: replay produced an out-of-range ancestor map; "
                       "skipping the proxy")
        return None
    is_collapsed = np.zeros(n_weld, dtype=bool)
    is_collapsed[np.asarray(collapses)[:, 1]] = True
    surv = np.full(len(pts_r), -1, dtype=np.int64)
    surv[vmap[~is_collapsed]] = np.where(~is_collapsed)[0]
    if (surv < 0).any():                      # replay/collapse bookkeeping mismatch
        logger.warning("build_skin_proxy: unmapped proxy vertices; skipping the proxy")
        return None
    return (np.asarray(pts_r, np.float64), np.asarray(faces_r, np.int64),
            vmap[weld])                       # raw vert -> proxy vert


def _face_normals(V: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unit face normals and areas. Degenerate faces get a zero normal (not a NaN)."""
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-30)[:, None], a * 0.5


def _repair_inverted_faces(pos: np.ndarray, faces: np.ndarray, anc_n: np.ndarray,
                           cand_start: np.ndarray, cand_order: np.ndarray, Pw: np.ndarray,
                           max_cand: int = 12, sweeps: int = 5) -> int:
    """Move the few vertices whose optimal position still inverts a face. Returns faces fixed.

    The simplifier's own flip test protects the mesh IN SIMPLIFY SPACE. Two things get past it:
    the importance warp is non-linear, so a large decimated triangle that is correctly oriented in
    warp space can invert once un-warped; and a later collapse can invalidate a decision an earlier
    one made. Both leave a handful of inverted faces, and because materials are single-sided an
    inverted face is an invisible one — a hole. They also CLUSTER, which is what makes them visible:
    38 inverted faces at 30k formed 32 separate patches, 9 of them over 10 mm².

    The repair is local and conservative. For each vertex touching an inverted face, try the
    positions of its own collapse-cluster members (nearest to the optimum first) and keep whichever
    inverts the fewest of its incident faces, breaking ties by staying closest to the optimum. Every
    candidate is a REAL SOURCE VERTEX, so a repaired vertex lands on the original surface rather
    than somewhere invented, and a vertex with no better option keeps the optimum it had.

    `anc_n` is the normal of the ORIGINAL face each output face descends from — the only correct
    reference for "inverted" (a nearest-face lookup mismatches exactly where the mesh is worst).
    """
    # Faces whose ancestor is degenerate have a meaningless reference normal — never chase those.
    gradable = np.linalg.norm(anc_n, axis=1) > 0.5
    # vertex -> incident faces, built once (CSR over the face corners)
    corner_v = faces.reshape(-1)
    corner_f = np.repeat(np.arange(len(faces)), 3)
    vorder = np.argsort(corner_v, kind="stable")
    v_start = np.searchsorted(corner_v[vorder], np.arange(len(pos) + 1))
    inc_f = corner_f[vorder]

    fixed = 0
    for _ in range(sweeps):
        n, _ = _face_normals(pos, faces)
        bad = gradable & ((n * anc_n).sum(1) < 0)
        if not bad.any():
            break
        before = int(bad.sum())
        for v in np.unique(faces[bad].reshape(-1)):
            inc = inc_f[v_start[v]:v_start[v + 1]]
            inc = inc[gradable[inc]]
            if len(inc) == 0:
                continue
            opt = pos[v].copy()
            cands = Pw[cand_order[cand_start[v]:cand_start[v + 1]][:max_cand]]
            best, best_bad, best_d = opt, None, 0.0
            for p in np.vstack([opt[None, :], cands]):
                pos[v] = p
                nn, _ = _face_normals(pos, faces[inc])
                cnt = int(((nn * anc_n[inc]).sum(1) < 0).sum())
                d = float(np.linalg.norm(p - opt))
                if best_bad is None or cnt < best_bad or (cnt == best_bad and d < best_d):
                    best, best_bad, best_d = p.copy(), cnt, d
            pos[v] = best
        n, _ = _face_normals(pos, faces)
        after = int((gradable & ((n * anc_n).sum(1) < 0)).sum())
        fixed += before - after
        if after >= before:      # no progress — further sweeps would only churn
            break
    return fixed


def _drop_zero_volume_flaps(pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Keep-mask that deletes coincident opposite-winding face pairs, when that cannot tear a hole.

    A collapse can fold the surface back onto itself, leaving two triangles on the same three
    vertices with opposite winding — a flap enclosing no volume. Measured on a 30k LOD: 48 such
    pairs, every one of them opposite-wound. They contribute nothing but z-fighting and the
    non-manifold edges that come with the pinch.

    Removal is only safe where it cannot leave a boundary edge. Deleting both faces drops each of
    their edges by two incidences, so an edge is fine at multiplicity exactly 2 (it disappears with
    the pair) or at 4+ (it stays manifold); at 3 it would be left with one face — an actual hole —
    and that pair is kept instead.
    """
    n_v = len(pos)
    tri = np.sort(faces, axis=1).astype(np.int64)
    # Row-wise unique, not a packed `(a*n + b)*n + c` key: that overflows int64 once the mesh has
    # more than ~2.1M vertices and then reports unrelated faces as coincident.
    _, inv_f, cnt_f = np.unique(tri, axis=0, return_inverse=True, return_counts=True)
    inv_f = inv_f.reshape(-1)
    if not (cnt_f[inv_f] > 1).any():
        return np.ones(len(faces), dtype=bool)

    e = np.concatenate([np.sort(faces[:, [0, 1]], 1), np.sort(faces[:, [1, 2]], 1),
                        np.sort(faces[:, [2, 0]], 1)])
    ekey = (e[:, 0].astype(np.int64) * n_v + e[:, 1]).reshape(3, -1).T
    _, inv_e, cnt_e = np.unique(ekey.reshape(-1), return_inverse=True, return_counts=True)
    mult = cnt_e[inv_e].reshape(-1, 3)

    keep = np.ones(len(faces), dtype=bool)
    normals, _ = _face_normals(pos, faces)
    for grp in np.where(cnt_f > 1)[0]:
        ids = np.where(inv_f == grp)[0]
        if len(ids) != 2:
            continue                                  # 3+ coincident faces: leave well alone
        a, b = ids
        if float(normals[a] @ normals[b]) > -0.5:
            continue                                  # same winding: a true duplicate, not a flap
        if not ((mult[a] == 2) | (mult[a] >= 4)).all():
            continue                                  # removing this pair would tear the surface
        keep[a] = keep[b] = False
    return keep


def _seam_fins(Pw32: np.ndarray, Fw: np.ndarray, Fraw: np.ndarray, attrs: Dict[str, np.ndarray],
               labels: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Constraint fins that make the simplifier respect ATTRIBUTE SEAMS. Returns (apex points,
    fin faces over welded ids + apex ids numbered from len(Pw32)).

    A UV seam, a hard edge or a material border is invisible to a position-only quadric wherever
    the surface is flat, so the collapse runs straight across it and the faces on one side end up
    painted with the other side's texels. On a character the seams sit on curved surfaces and this
    stayed rare; on a kit piece — a flat wall whose decal sits in its own atlas chart — it smeared
    the decal across the wall (measured 638 texels of UV drift at p95 on a real entrance at 25%).

    The textbook remedy is a constraint plane through each seam edge, perpendicular to the surface.
    fast_simplification has no hook for one, so it is given as GEOMETRY: one extra triangle per
    seam edge, standing on the edge along the surface normal. Its plane is exactly that constraint,
    and because its two new edges belong to a single triangle the seam's vertices become BORDER
    vertices, which the simplifier only collapses into other border vertices — i.e. along the seam,
    never across it. The fins are dropped after the replay; no output face comes from them.
    """
    keys = []
    for name in ("TEXCOORD_0", "NORMAL"):
        a = attrs.get(name)
        if a is not None:
            keys.append(np.round(np.asarray(a, dtype=np.float64).reshape(len(a), -1), 4))
    if labels is not None:
        keys.append(np.asarray(labels, dtype=np.float64)[:, None])
    if not keys:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    _, akey = np.unique(np.concatenate(keys, axis=1), axis=0, return_inverse=True)
    akey = akey.reshape(-1)
    # half-edges of the welded mesh with the raw vertex at each end
    wa = np.concatenate([Fw[:, 0], Fw[:, 1], Fw[:, 2]])
    wb = np.concatenate([Fw[:, 1], Fw[:, 2], Fw[:, 0]])
    ra = np.concatenate([Fraw[:, 0], Fraw[:, 1], Fraw[:, 2]])
    rb = np.concatenate([Fraw[:, 1], Fraw[:, 2], Fraw[:, 0]])
    fid = np.tile(np.arange(len(Fw)), 3)
    swap = wa > wb
    lo, hi = np.where(swap, wb, wa), np.where(swap, wa, wb)
    klo, khi = akey[np.where(swap, rb, ra)], akey[np.where(swap, ra, rb)]
    code = lo.astype(np.int64) * (int(Fw.max()) + 1) + hi
    order = np.argsort(code, kind="stable")
    code, klo, khi, fid, lo, hi = code[order], klo[order], khi[order], fid[order], lo[order], hi[order]
    starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
    cnt = np.diff(np.r_[starts, len(code)])
    seam = ((np.minimum.reduceat(klo, starts) != np.maximum.reduceat(klo, starts)) |
            (np.minimum.reduceat(khi, starts) != np.maximum.reduceat(khi, starts))) & (cnt >= 2)
    if not seam.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    P = Pw32.astype(np.float64)
    fn = np.cross(P[Fw[:, 1]] - P[Fw[:, 0]], P[Fw[:, 2]] - P[Fw[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-30
    nsum = np.add.reduceat(fn[fid], starts, axis=0)[seam]
    a, b = lo[starts][seam], hi[starts][seam]
    ln = np.linalg.norm(nsum, axis=1)
    elen = np.linalg.norm(P[b] - P[a], axis=1)
    ok = (ln > 1e-6) & (elen > 0)
    a, b, n, elen = a[ok], b[ok], nsum[ok] / ln[ok, None], elen[ok]
    apex = (0.5 * (P[a] + P[b]) + n * elen[:, None]).astype(np.float32)
    ids = len(Pw32) + np.arange(len(a))
    return apex, np.stack([a, b, ids], axis=1).astype(np.int64)


def _decimate_primitive(
    P: np.ndarray,
    F: np.ndarray,
    attrs: Dict[str, np.ndarray],
    target_ratio: float,
    stats: Optional[Dict] = None,
    head_boost: Optional[float] = None,
    hand_boost: Optional[float] = None,
    return_source: bool = False,
    seam_lock: bool = False,
    labels: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, ...]]:
    """Seam-preserving decimation of one primitive. Returns (positions, faces, attrs) or None
    if the primitive is too small / the reduction is a no-op.

    `stats`, when given, receives density telemetry (`head_vertex_share_before/after`).
    `return_source=True` appends a 4th array: the raw source vertex each output vertex copied its
    attributes from. `_decimate_glb` uses it to split a jointly decimated multi-primitive mesh back
    into its primitives (a raw vertex belongs to exactly one of them).
    `seam_lock` keeps collapses from crossing attribute seams (see `_seam_fins`); `labels` (one per
    raw vertex, e.g. the primitive it came from) adds borders the attributes alone do not show.
    """
    import fast_simplification as _fs

    n_raw = len(P)
    if n_raw < 64 or len(F) < 64:
        return None

    weld, n_weld = _weld(P)
    # representative position per welded vertex (first raw occurrence)
    order = np.argsort(weld, kind="stable")
    weld_sorted = weld[order]
    starts = np.searchsorted(weld_sorted, np.arange(n_weld))
    counts = np.diff(np.append(starts, len(order)))
    Pw = P[order[starts]].astype(np.float64)
    Fw = weld[F]
    # drop faces degenerate under welding (zero-area slivers across seams)
    nz = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])
    Fw_clean = Fw[nz]
    F_clean = F[nz]
    if len(Fw_clean) < 64:
        return None

    # Simplify in a normalized, importance-warped space (see _SIMPLIFY_SPAN / _importance_warp).
    # Both transforms are invertible, and the merged positions the simplifier computes here are
    # mapped back to real space before they are written (see `pos_out` below).
    Qw, head_mask, warp_steps = _importance_warp(Pw, head_boost, hand_boost)
    span = float(np.linalg.norm(Qw.max(0) - Qw.min(0)))
    unit = _SIMPLIFY_SPAN / span if span > 0 else 1.0
    Pw32 = (Qw * unit).astype(np.float32)   # replay requires float32 points
    Fw32 = Fw_clean.astype(np.int32)
    # target_count is exact where target_reduction rounds; the simplifier may still stop short of
    # it when the schedule runs out of collapsible edges, which the caller reports as-is.
    target_count = int(max(16, round(len(Fw32) * float(np.clip(target_ratio, 0.02, 0.95)))))
    apex, fins = (_seam_fins(Pw32, Fw_clean, F_clean, attrs, labels) if seam_lock
                  else (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)))
    P_in = np.concatenate([Pw32, apex]) if len(fins) else Pw32
    F_in = np.concatenate([Fw32, fins.astype(np.int32)]) if len(fins) else Fw32
    # The simplifier counts the fins' faces toward its target, but most fins collapse away with
    # their seam edges, so a target of "real + fins" lands that many real faces OVER. Measure the
    # real faces the replay actually kept and run once more with the target corrected.
    goal = target_count + len(fins)
    for _attempt in range(3 if len(fins) else 1):
        out = _fs.simplify(P_in, F_in, target_count=int(max(goal, 16)), agg=_SIMPLIFY_AGG,
                           return_collapses=True)
        _, _, collapses = out
        if collapses is None or len(collapses) == 0:
            return None
        pts_r, _, vmap = _fs.replay_simplification(P_in, F_in, collapses)
        vmap = np.asarray(vmap, dtype=np.int64)
        if not len(fins):
            break
        Fr = vmap[Fw_clean]
        real = int(((Fr[:, 0] != Fr[:, 1]) & (Fr[:, 1] != Fr[:, 2]) & (Fr[:, 0] != Fr[:, 2])).sum())
        if real <= target_count * 1.05 or goal <= 16:
            break
        goal -= real - target_count
    pts_r = np.asarray(pts_r, dtype=np.float64)
    if len(fins):
        # Drop the fins' apex points: keep only output vertices some REAL vertex maps to.
        has_real = np.bincount(vmap[:n_weld], minlength=len(pts_r)) > 0
        remap = np.cumsum(has_real) - 1
        vmap = remap[vmap[:n_weld]]
        pts_r = pts_r[has_real]
        collapses = np.asarray(collapses)
        collapses = collapses[collapses[:, 1] < n_weld]
        if stats is not None:
            stats["seam_fins"] = int(len(fins))

    # THE MERGED POSITION IS NOT NEGOTIABLE. `pts_r` is the quadric-optimal position of each
    # collapse cluster, and it is the position the simplifier's own normal-flip test
    # (`Simplify::flipped`) was evaluated at when it accepted the collapse. Emitting the surviving
    # ORIGINAL vertex instead — which this did until 2026-09-05 — moves the vertex a median 4.5 mm
    # (max 139 mm at 30k) away from where that test passed, and the faces around it fold over:
    # measured 1.75% / 3.28% / 4.07% of faces inverted at 100k / 30k / 15k tris, which is 0.00% at
    # every rung once the optimal position is used. Materials are single-sided, so an inverted face
    # is an invisible one — that was the reported "holes and floating shards" in the LODs.
    # A vertex that absorbed nobody is returned unchanged (measured exactly 0.0 mm), so this only
    # moves vertices that actually merged.
    opt_real = _inverse_importance_warp(pts_r / unit, warp_steps)

    # surviving original welded vertex behind each output vertex
    is_collapsed = np.zeros(n_weld, dtype=bool)
    is_collapsed[np.asarray(collapses)[:, 1]] = True
    surv_ids = np.where(~is_collapsed)[0]
    surv = np.full(len(pts_r), -1, dtype=np.int64)
    surv[vmap[surv_ids]] = surv_ids
    # (with seam fins, a cluster whose survivor was an apex point has no real survivor — expected)
    if (surv < 0).any() and not len(fins):   # replay bookkeeping mismatch — bail, never corrupt
        logger.warning("retopotool: unmapped output vertices; skipping primitive")
        return None
    if stats is not None and head_mask.any():
        stats["head_vertex_share_before"] = float(head_mask.mean())
        stats["head_vertex_share_after"] = float(head_mask[surv_ids].mean())

    # ATTRIBUTE ANCHOR: the welded vertex of each collapse cluster NEAREST the position actually
    # emitted. `surv` is not that vertex — it is merely whichever index the simplifier chose to keep,
    # and once the vertex moves to the cluster optimum it is the nearest member for only ~24% of
    # output vertices (measured at 30k). Copying UVs and skin from a vertex several centimetres away
    # is how a knee weight lands on a shin: the DOMINANT BONE differed for 2.5-3.6% of vertices.
    # Distances are taken in simplify space, the metric the collapse decisions themselves used.
    d_opt = np.linalg.norm(Pw32.astype(np.float64) - pts_r[vmap], axis=1)
    _o = np.lexsort((d_opt, vmap))                      # group by cluster, nearest first inside it
    _cand_start = np.searchsorted(vmap[_o], np.arange(len(pts_r) + 1))
    anchor = _o[_cand_start[:-1]]
    # A cluster of ONE absorbed nobody, so its vertex never moved — emit the source position itself.
    # Read back through the float32 simplify space it is off by a rounding step, which is enough to
    # break bit-exact seams with geometry that was not decimated (another mesh, a skipped part).
    single = np.diff(_cand_start) == 1
    opt_real[single] = Pw[anchor[single]]
    if stats is not None:
        stats["anchor_is_survivor_frac"] = float(np.mean(anchor == surv))

    # remap raw faces -> output welded ids, drop degenerates
    Fo = vmap[Fw_clean]
    keep = (Fo[:, 0] != Fo[:, 1]) & (Fo[:, 1] != Fo[:, 2]) & (Fo[:, 0] != Fo[:, 2])
    Fo = Fo[keep]
    Fraw = F_clean[keep]
    if len(Fo) < 16:
        return None

    # THE REFERENCE ORIENTATION, captured BEFORE anything retriangulates. `anc_n` is the normal of
    # the ORIGINAL source face each output face descends from, and the solidity pass below grades
    # every output face against it. It must be computed here, while `Fraw` still holds real source
    # faces: the quality pass reassembles `Fraw` corner-by-corner, so a flipped face's raw triple is
    # no longer a triangle that ever existed and its "ancestor normal" would be meaningless. Reading
    # it after the flips made the solidity pass mistake 273 perfectly good faces for inverted ones
    # and reverse their winding, which is 3x that many edges where neighbours disagree on which side
    # is out — 811 on a 30k rung, up from 3.
    anc_faces = np.where(nz)[0][keep]     # ORIGINAL face index behind each output face
    anc_n, _ = _face_normals(P.astype(np.float64), Fraw)

    # ---- triangle-quality pass (flips only; no vertex moves) ------------------------------------
    # Runs BEFORE the solidity pass so solidity stays the last word on face validity, and after the
    # face remap so it operates on the faces that will actually be written.
    # ISLANDS ARE COMPUTED WITH OR WITHOUT UVs. An island is a connected component of the RAW face
    # graph, which splits not only at UV seams but at every hard (split-normal) edge and between the
    # primitives of a jointly decimated mesh. Both the flip pass and the wedge re-split below must
    # stay inside one: a flip across a hard edge rounds off a crate's corner, and a wedge that picks
    # a raw vertex from the far side takes that side's NORMAL (or another primitive's attributes).
    isl = _uv_islands(n_raw, F)
    n_flips = 0
    if _QUALITY_PASS and len(Fo) >= 64:
        crease_cos = float(np.cos(np.radians(_FLIP_CREASE_DEG)))
        for _ in range(_FLIP_ROUNDS):
            got = _flip_round(opt_real, Fo, Fraw, anc_n, isl, crease_cos, _FLIP_MIN_GAIN_DEG)
            n_flips += got
            if got == 0:
                break

    # ---- solidity pass -------------------------------------------------------------------------
    # The simplifier only guaranteed these faces in SIMPLIFY space, and it applies no topology test
    # at all. Three cheap repairs, in order, each measured on a real character at 30k:
    #   1. re-place the vertices whose optimum inverts a face   (44 inverted -> 7)
    #   2. delete zero-volume flaps                             (48 pairs, all opposite-wound)
    #   3. flip whatever is still inverted                      (last resort, see below)
    n_fix = _repair_inverted_faces(opt_real, Fo, anc_n, _cand_start, _o, Pw)

    flap_keep = _drop_zero_volume_flaps(opt_real, Fo)
    n_flaps = int((~flap_keep).sum())
    if n_flaps:
        Fo, Fraw, anc_n = Fo[flap_keep], Fraw[flap_keep], anc_n[flap_keep]
        anc_faces = anc_faces[flap_keep]

    # Anything still inverted has a fold no single vertex position can undo. Left alone it is
    # BACKFACE-CULLED — an actual see-through hole — so the winding is flipped instead: the face
    # renders solid, and its shading comes from the smooth vertex normals the bake recomputes, not
    # from this triangle. A sub-centimetre shading error beats a hole. Both `Fo` and `Fraw` flip
    # together or the per-corner attribute gather below would read the wrong ancestor.
    out_n, out_a = _face_normals(opt_real, Fo)
    gradable = np.linalg.norm(anc_n, axis=1) > 0.5
    still = gradable & ((out_n * anc_n).sum(1) < 0)
    if still.any():
        Fo[still] = Fo[still][:, ::-1]
        Fraw[still] = Fraw[still][:, ::-1]
    if stats is not None:
        # Which ORIGINAL face each output face descends from — what lets a test compare an output
        # normal against the normal it should still have (a nearest-face lookup mismatches exactly
        # where the mesh is worst). Note the flipped faces read as CORRECT through this map, which
        # is what the renderer sees; `winding_flipped` is the honest count of the folds behind it.
        stats["ancestor_faces"] = anc_faces
        stats["inverted_faces_repaired"] = int(n_fix)
        stats["zero_volume_flaps_dropped"] = n_flaps
        stats["winding_flipped"] = int(still.sum())
        stats["quality_flips"] = int(n_flips)
        stats["inverted_area_frac"] = float(out_a[still].sum() / max(out_a.sum(), 1e-30))

    # candidate raw verts per welded vertex (seam splits), padded with the first candidate
    kmax = int(min(max(int(counts.max()), 1), _KMAX))
    cand = np.empty((n_weld, kmax), dtype=np.int64)
    for k in range(kmax):
        cand[:, k] = order[starts + np.minimum(k, counts - 1)]

    w_corners = Fo.reshape(-1)                       # (C,) output vert id per corner
    anc_corners = Fraw.reshape(-1)                   # (C,) ancestor raw vert per corner
    cc = cand[anchor[w_corners]]                     # (C, kmax) candidate raw verts

    uv = attrs.get("TEXCOORD_0")
    # per-corner: candidate on the ANCESTOR'S ISLAND whose UV is nearest the ancestor's (chunked to
    # bound memory). Cross-island candidates are excluded outright — their UVs point at unrelated
    # atlas regions (or their normals at the other side of a hard edge), and picking one smears that
    # across the face. When the anchor vertex has no split on the ancestor's island (the collapse
    # crossed a seam), keep the ancestor's own attributes at the new position: the corner then
    # samples its original texels instead of another island's. Without UVs the same island rule
    # applies with no distance to rank by.
    uvf = uv.astype(np.float32) if uv is not None else None
    anc_isl = isl[anc_corners]
    chosen = np.empty(len(cc), dtype=np.int64)
    CHUNK = 500_000
    for s in range(0, len(cc), CHUNK):
        e = min(s + CHUNK, len(cc))
        block = cc[s:e]
        if uvf is not None:
            d = uvf[block] - uvf[anc_corners[s:e]][:, None, :]
            dist = (d * d).sum(-1)
        else:
            dist = np.zeros(block.shape, dtype=np.float32)
        same = isl[block] == anc_isl[s:e][:, None]
        dist[~same] = np.inf
        pick = block[np.arange(e - s), np.argmin(dist, axis=1)]
        chosen[s:e] = np.where(same.any(axis=1), pick, anc_corners[s:e])

    # dedupe (output welded vert, chosen raw ancestor) -> final split vertices
    key = w_corners * np.int64(n_raw) + chosen
    uniq_key, inv = np.unique(key, return_inverse=True)
    Fo_new = inv.reshape(-1, 3).astype(np.int64)
    vw = uniq_key // n_raw
    vr = uniq_key % n_raw
    # One position per OUTPUT WELDED vertex (`vw`), so every UV-split copy of it lands on the same
    # point and the seam stays welded — the whole reason this module simplifies welded in the first
    # place. `opt_real` is already back in real space (see `_inverse_importance_warp`).
    pos_out = opt_real[vw].astype(np.float32)

    # SKIN IS A FUNCTION OF POSITION, NOT OF THE UV WEDGE. Every other attribute is correctly keyed
    # on `vr` (the per-corner raw ancestor) — that exact copy is precisely what keeps a UV seam's two
    # sides pointing at their own atlas regions. Skin is different: the two sides of a seam are the
    # SAME POINT ON THE BODY, so if they take weights from two different raw vertices they deform
    # apart and the seam rips open. Keying skin on `vr` did exactly that, and because the split
    # copies are duplicate vertices rather than neighbours across an edge, every static check the
    # pipeline runs (boundary edges, winding, volume, point-to-surface) reports a perfect mesh.
    # MEASURED on a real character — weld groups whose members disagree on their weights, and how
    # far those members travel apart under a test pose:
    #     rig 400k   0.3% of groups,  posed seam gap p95 0.00 mm
    #     LOD 100k  42.3%,            p95  1.70 mm
    #     LOD  30k  60.3%,            p95  5.06 mm, p99 15.0 mm
    #     LOD  15k  65.8%,            p95  8.94 mm, p99 25.8 mm, max L1 2.000 (a TOTAL bone swap
    #                                 between two vertices at the identical position)
    # Regionally at 30k: head 5.6% of groups, torso 71.3%, legs 68.0% — which is exactly the
    # reported "many holes on the body, the face is working pretty well". A centimetre-wide gap
    # along a seam in a posed character IS the hole.
    # The fix is to resolve skin ONCE PER OUTPUT WELDED VERTEX (`vw`) so every wedge sharing that
    # position also shares its weights. `_skin_source_raw` is the anchor's own raw vertex, so the
    # emitted row is still a verbatim copy of one real source vertex: weights already sum to 1, and
    # a collapse still cannot invent an influence.
    attrs_out = {name: arr[vr] for name, arr in attrs.items() if name not in _SKIN_NAMES}

    # UV IS A FUNCTION OF POSITION, and the position just moved. `arr[vr]` above gave this wedge
    # the UV of a real source vertex — correct for choosing its atlas chart, wrong as a value once
    # the vertex is no longer standing there. Re-read it from the source surface at the emitted
    # position, inside the chart the copy just picked. See `_resample_uv_on_source` / _UV_RESAMPLE.
    if _UV_RESAMPLE and uv is not None:
        uv_sets = {n: a for n, a in attrs.items()
                   if n.startswith("TEXCOORD_") and a is not None and a.ndim == 2
                   and np.issubdtype(a.dtype, np.floating)}
        try:
            attrs_out.update(_resample_uv_on_source(
                P.astype(np.float64), F, uv_sets, vr, opt_real[vw], stats=stats))
        except Exception:
            logger.warning("retopotool: UV resample failed; keeping the verbatim copy",
                           exc_info=True)
    skin_names = [n for n in attrs if n in _SKIN_NAMES]
    if skin_names:
        skin_src_raw = order[starts[anchor]]          # output vert -> one raw vertex, seam-consistent
        blended = None
        # Blending is only defined for the 4-influence pair. An 8-influence rig (JOINTS_1) would need
        # the two pairs merged and re-split; this pipeline never produces one, so take the verbatim
        # path rather than guess.
        if (_SKIN_BLEND and "JOINTS_0" in attrs and "WEIGHTS_0" in attrs
                and "JOINTS_1" not in attrs):
            try:
                rep = order[starts]                   # welded vertex -> its raw representative
                J_w = attrs["JOINTS_0"][rep].astype(np.int64)
                W_w = attrs["WEIGHTS_0"][rep].astype(np.float64)
                W_w = W_w / np.maximum(W_w.sum(1, keepdims=True), 1e-12)
                # incident-area share per welded vertex, on the welded triangles the collapse saw
                _, fa = _face_normals(Pw, Fw_clean)
                mass = np.zeros(n_weld, dtype=np.float64)
                for k in range(3):
                    np.add.at(mass, Fw_clean[:, k], fa / 3.0)
                bj, bw = _blend_skin_over_clusters(J_w, W_w, mass, d_opt, vmap, _cand_start,
                                                   _o, anchor, len(pts_r))
                blended = {"JOINTS_0": bj, "WEIGHTS_0": bw}
            except Exception:
                logger.warning("retopotool: skin blend failed; using the verbatim anchor row",
                               exc_info=True)
                blended = None
        for name in skin_names:
            if blended is not None and name in blended:
                attrs_out[name] = blended[name][vw]
            else:
                attrs_out[name] = attrs[name][skin_src_raw[vw]]
        if stats is not None:
            stats["skin_keyed_on_position"] = True
            stats["skin_blended"] = blended is not None
    if return_source:
        return pos_out, Fo_new, attrs_out, vr
    return pos_out, Fo_new, attrs_out


def _incident_faces(n_raw: int, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Source faces touching each raw vertex, as CSR: (faces, start) with `faces[start[v]:start[v+1]]`.

    int32 because the pipeline rejects meshes past `MAX_TOTAL_VERTS`, and CSR rather than a padded
    (n_raw, cap) table because the pad has to be as wide as the worst vertex on the mesh — see
    `_UV_GATHER_MAX` for why capping it instead is not free.
    """
    vid = F.ravel()
    order = np.argsort(vid, kind="stable")
    faces = np.repeat(np.arange(len(F), dtype=np.int32), 3)[order]
    return faces, np.searchsorted(vid[order], np.arange(n_raw + 1))


def _gather_incident(faces: np.ndarray, start: np.ndarray, verts: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """(m, w) face ids incident to each of `verts` and a validity mask, w = this batch's own
    widest degree. Out-of-range columns read a harmless in-bounds slot and are masked off."""
    st = start[verts]
    deg = np.minimum(start[verts + 1] - st, _UV_GATHER_MAX)
    cols = np.arange(int(deg.max()) if len(deg) else 0)
    ok = cols[None, :] < deg[:, None]
    idx = np.minimum(st[:, None] + cols[None, :], len(faces) - 1)
    return faces[idx], ok


def _vertex_ring(n_raw: int, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Raw vertices sharing a face edge with each raw vertex, as CSR.

    Edges are de-duplicated through a single packed int64 key rather than `np.unique(..., axis=0)`:
    a 3M-face mesh has 18M directed edges, and the row-wise unique lexsorts a 288 MB array to
    produce the same answer.
    """
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0).astype(np.int64)
    e = np.concatenate([e, e[:, ::-1]], axis=0)
    packed = np.unique(e[:, 0] * np.int64(n_raw) + e[:, 1])
    return (packed % n_raw).astype(np.int32), np.searchsorted(packed // n_raw, np.arange(n_raw + 1))


def _closest_on_triangles(pts: np.ndarray, A: np.ndarray, E1: np.ndarray, E2: np.ndarray,
                          d00: np.ndarray, d01: np.ndarray, d11: np.ndarray, den: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Squared distance and barycentric coordinates of the closest point on each triangle.

    The interior case is the plane projection; OUTSIDE the triangle the answer is the best of the
    three edge projections, which is not what clipping the barycentric pair and renormalizing it
    gives. Renormalizing (`v, w -> v/(v+w), w/(v+w)`) slides along the ray from A, and that only
    coincides with the perpendicular foot on BC for special triangles — so it both mis-ranks
    candidate faces and hands back a UV read at the wrong place. It matters precisely while the walk
    is still approaching, where every candidate is an exterior one.
    """
    q = pts - A
    d20 = (q * E1).sum(-1)
    d21 = (q * E2).sum(-1)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    inside = (v >= 0.0) & (w >= 0.0) & (v + w <= 1.0)

    def _edge(o, e, dee):
        t = np.clip((( pts - o) * e).sum(-1) / np.maximum(dee, 1e-30), 0.0, 1.0)
        p = o + t[..., None] * e
        return t, ((p - pts) ** 2).sum(-1)

    E3 = E2 - E1                                          # B -> C
    tab, dab = _edge(A, E1, d00)                          # A -> B
    tac, dac = _edge(A, E2, d11)                          # A -> C
    tbc, dbc = _edge(A + E1, E3, (E3 * E3).sum(-1))       # B -> C
    bv = np.where(dab <= dac, tab, 0.0)
    bw = np.where(dab <= dac, 0.0, tac)
    bd = np.minimum(dab, dac)
    use_bc = dbc < bd
    bv = np.where(use_bc, 1.0 - tbc, bv)
    bw = np.where(use_bc, tbc, bw)
    bd = np.where(use_bc, dbc, bd)

    v = np.where(inside, v, bv)
    w = np.where(inside, w, bw)
    p_in = A + v[..., None] * E1 + w[..., None] * E2
    d2 = np.where(inside, ((p_in - pts) ** 2).sum(-1), bd)
    return d2, np.stack([1.0 - v - w, v, w], axis=-1)


def _resample_uv_on_source(P: np.ndarray, F: np.ndarray, uv_sets: Dict[str, np.ndarray],
                           vr: np.ndarray, pos: np.ndarray, stats: Optional[Dict] = None
                           ) -> Dict[str, np.ndarray]:
    """UV of each output wedge, read off the SOURCE surface at the position actually emitted.

    `vr[i]` is the wedge's ancestor raw vertex and `pos[i]` the emitted position. The wedge's UV
    becomes the source parameterization's value at `pos[i]`, located by walking the source mesh from
    the ancestor (see _UV_RESAMPLE) — so a seam's two sides still read their own atlas charts, which
    is the invariant the verbatim copy existed to protect, while neither of them slides.

    The walk is a MONOTONE descent: a wedge only re-centres when the round strictly improved its
    best squared distance, so the seed sequence cannot cycle and the pass terminates on its own —
    `_UV_RESAMPLE_ROUNDS` is a budget, not the thing that makes it stop. A wedge keeps the nearest
    point it reached, which is still inside its own chart. Read `uv_offsurface_*` (metres) as the
    quality signal, NOT `uv_resample_unconverged`: the latter only counts wedges still moving when
    the budget ran out, while the former is how far off the surface the answer actually landed.

    Every UV set is evaluated with the SAME face and barycentric coordinates, which is what keeps a
    second set registered against the first.
    """
    n_raw = len(P)
    inc_faces, inc_start = _incident_faces(n_raw, F)
    ring = None

    A = P[F[:, 0]]
    E1 = P[F[:, 1]] - A
    E2 = P[F[:, 2]] - A
    d00 = (E1 * E1).sum(1)
    d01 = (E1 * E2).sum(1)
    d11 = (E2 * E2).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)      # degenerate source triangle

    n = len(pos)
    best_f = np.full(n, -1, dtype=np.int64)
    best_b = np.zeros((n, 3))
    best_d = np.full(n, np.inf)
    seed = vr.astype(np.int64).copy()

    def _walk(active: np.ndarray, max_rounds: int, two_ring: bool) -> Tuple[int, np.ndarray]:
        chunk = _UV_RESAMPLE_CHUNK // 8 if two_ring else _UV_RESAMPLE_CHUNK
        nonlocal ring
        if two_ring and ring is None:
            ring = _vertex_ring(n_raw, F)
        rounds = 0
        for _ in range(max(1, max_rounds)):
            rounds += 1
            keep = np.zeros(len(active), dtype=bool)
            for lo in range(0, len(active), chunk):
                sub = active[lo:lo + chunk]
                rows = np.arange(len(sub))
                cand, ok = _gather_incident(inc_faces, inc_start, seed[sub])
                if two_ring:
                    nb, nb_ok = _gather_incident(ring[0], ring[1], seed[sub])
                    flat = np.where(nb_ok, nb, 0).ravel().astype(np.int64)
                    ex, ex_ok = _gather_incident(inc_faces, inc_start, flat)
                    ex = ex.reshape(len(sub), -1)
                    ex_ok = ex_ok.reshape(len(sub), -1) & np.repeat(nb_ok, ex.shape[1] // nb.shape[1],
                                                                    axis=1)
                    cand = np.concatenate([cand, ex], axis=1)
                    ok = np.concatenate([ok, ex_ok], axis=1)
                cs = np.where(ok, cand, 0).astype(np.int64)
                pts = pos[sub][:, None, :]
                d2, bary = _closest_on_triangles(pts, A[cs], E1[cs], E2[cs],
                                                 d00[cs], d01[cs], d11[cs], den[cs])
                d2 = np.where(ok, d2, np.inf)
                pick = np.argmin(d2, axis=1)
                dmin = d2[rows, pick]
                # STRICT improvement, or this wedge is done: that is what forbids a two-seed cycle.
                take = dmin < best_d[sub]
                idx = sub[take]
                best_d[idx] = dmin[take]
                best_f[idx] = cs[rows, pick][take]
                best_b[idx] = bary[rows, pick][take]
                # step: re-centre on the corner of the winning face nearest the emitted position
                tri = F[cs[rows, pick]]
                corner = np.argmin(((P[tri] - pts) ** 2).sum(-1), axis=1)
                nxt = tri[rows, corner]
                keep[lo:lo + len(sub)] = take & (nxt != seed[sub])
                seed[sub] = np.where(take, nxt, seed[sub])
            active = active[keep]
            if not len(active):
                break
        return rounds, active

    rounds, active = _walk(np.arange(n), _UV_RESAMPLE_ROUNDS, two_ring=False)
    # Wedges whose closest point sits on a triangle BOUNDARY are not seated — the true face is
    # outside the patch the one-ring walk could see. Spend the wide neighbourhood on those only.
    stalled = np.flatnonzero((best_b.min(1) <= 1e-9) & (best_d > 0.0))
    esc_rounds = 0
    if len(stalled) and _UV_ESCALATE_ROUNDS > 0:
        esc_rounds, esc_active = _walk(stalled, _UV_ESCALATE_ROUNDS, two_ring=True)
        active = np.union1d(active, esc_active)

    out = {k: v[vr].copy() for k, v in uv_sets.items()}
    hit = best_f >= 0
    tri = F[np.where(hit, best_f, 0)]
    for name, uv in uv_sets.items():
        new = (best_b[:, 0:1] * uv[tri[:, 0]] + best_b[:, 1:2] * uv[tri[:, 1]]
               + best_b[:, 2:3] * uv[tri[:, 2]])
        out[name] = np.where(hit[:, None], new, out[name]).astype(uv.dtype)
    if stats is not None:
        off = np.sqrt(np.where(hit, best_d, 0.0))
        stats["uv_resampled"] = True
        stats["uv_resample_fallback"] = int((~hit).sum())
        stats["uv_resample_rounds"] = rounds
        stats["uv_escalated"] = int(len(stalled))
        stats["uv_escalate_rounds"] = esc_rounds
        stats["uv_resample_unconverged"] = int(len(active))
        stats["uv_offsurface_p95"] = float(np.percentile(off, 95)) if n else 0.0
        stats["uv_offsurface_max"] = float(off.max()) if n else 0.0
    return out


def _texture_image_index(g: GLTF2, tex_index: Optional[int]) -> Optional[int]:
    """Image behind a texture, honouring EXT_texture_webp (whose source lives in the extension
    block, not the core `source` slot — reading only the core slot silently misses WebP media)."""
    if tex_index is None or not g.textures or tex_index >= len(g.textures):
        return None
    tex = g.textures[tex_index]
    if tex.source is not None:
        return tex.source
    ext = (tex.extensions or {}).get("EXT_texture_webp") or {}
    src = ext.get("source")
    return int(src) if src is not None else None


def _reference_highpoly(path: str) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """Load an external mesh to bake FROM, keyed by (mesh index, primitive index). Used by the LOD
    ladder, where the high-poly is the full-resolution source rather than the file being decimated.
    Each decimated primitive bakes against the reference primitive at the SAME key — taking one
    primitive for all of them (what this did while every input was a one-primitive character)
    bakes a multi-material prop's every part against its first part."""
    g = GLTF2().load(path)
    blob = g.binary_blob()
    if blob is None:
        # The file parsed as a JSON glTF, not a GLB — see the extension note at the save site.
        raise ValueError(f"{path} has no binary chunk; it was not written as a GLB")
    from . import bake_normals as _b
    out: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    for mi, mesh in enumerate(g.meshes or []):
        for pi, prim in enumerate(mesh.primitives):
            at = prim.attributes
            if at.POSITION is None or at.TEXCOORD_0 is None or (prim.mode or 4) != 4:
                continue
            P = _read_position(g, blob, at.POSITION)
            F = (_acc(g, blob, prim.indices).astype(np.int64).reshape(-1, 3)
                 if prim.indices is not None else np.arange(len(P), dtype=np.int64).reshape(-1, 3))
            mat = (g.materials[prim.material]
                   if prim.material is not None and g.materials else None)
            nmap = None
            if mat is not None and mat.normalTexture is not None:
                raw = _image_bytes(g, blob, _texture_image_index(g, mat.normalTexture.index))
                nmap = _b.decode_image(raw) if raw is not None else None
            out[(mi, pi)] = {
                "P": P.astype(np.float64), "F": F,
                "UV": _read_float_attr(g, blob, "TEXCOORD_0", at.TEXCOORD_0).astype(np.float64),
                "NORMAL": (_read_float_attr(g, blob, "NORMAL", at.NORMAL).astype(np.float64)
                           if at.NORMAL is not None else None),
                "TANGENT": (_read_float_attr(g, blob, "TANGENT", at.TANGENT).astype(np.float64)
                            if at.TANGENT is not None else None),
                "normal_map": nmap,
            }
    return out


def _image_bytes(g: GLTF2, blob: bytes, img_idx: Optional[int]) -> Optional[bytes]:
    """Embedded bytes of an image (None for external / data-URI images or a bad index)."""
    if img_idx is None or not g.images or img_idx >= len(g.images):
        return None
    img = g.images[img_idx]
    if img.bufferView is None:
        return None
    bv = g.bufferViews[img.bufferView]
    off = bv.byteOffset or 0
    return bytes(blob[off:off + bv.byteLength])


def _image_size(raw: Optional[bytes]) -> Optional[int]:
    """Largest dimension of an encoded image, from its header only."""
    if not raw:
        return None
    try:
        from PIL import Image as _PIL
        import io as _io
        with _PIL.open(_io.BytesIO(raw)) as im:
            return int(max(im.size))
    except Exception:
        return None


def _uv_overlap_frac(uvs: List[np.ndarray], faces: List[np.ndarray], res: int = 512) -> float:
    """How much of the UV layout is claimed twice: 1 - (texels covered) / (texels the triangles'
    UV area adds up to). Sampling texel centres is unbiased for area, so an atlas whose charts do
    not overlap lands near 0 at any density; a mirrored or stacked layout lands near the fraction
    of area it reuses."""
    from .bake_normals import _rasterize_uv
    covered = np.zeros((res, res), dtype=bool)
    area = 0.0
    for uv, F in zip(uvs, faces):
        uv = np.asarray(uv, dtype=np.float64)
        d1 = uv[F[:, 1]] - uv[F[:, 0]]
        d2 = uv[F[:, 2]] - uv[F[:, 0]]
        area += float(np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]).sum() * 0.5)
        tri_id, _ = _rasterize_uv(uv, F, res)
        covered |= tri_id >= 0
    expect = area * res * res
    if expect < 64:                         # too little UV area to judge; do not block the bake
        return 0.0
    return float(max(0.0, 1.0 - covered.sum() / expect))


# The bake is refused for a material whose UV layout reuses more than this share of its area
# (mirrored halves, stacked islands): a per-texel normal map cannot hold two different surfaces.
BAKE_MAX_UV_OVERLAP = 0.10
# A baked map is KEPT only if it brings the shading closer to the source than the decimated mesh
# with its carried normals and original map already is (`_bake_helps`). Measured on real kit props
# at 25% (mean angular error vs the source, baked / unbaked): a barrel 6.7 / 16.0 deg, a tank
# 3.7 / 12.0, a cement entrance 6.0 / 8.4 — the bake repairs what the decimation broke — but a
# door 15.0 / 14.8, whose relief already lived in its normal map: re-encoding that map costs more
# than the decimation did, so it keeps its original map and normals.
# The statistic is a TRIMMED mean: samples where either variant is off by more than
# BAKE_CHECK_TRIM_DEG are dropped. On double-sided kit geometry (walls with back-to-back faces)
# the nearest-surface lookup lands on the wrong side for up to 20% of samples, at ~140 deg for
# both variants, and a plain mean drowns the real difference in that noise. A median is wrong the
# other way: it ignores exactly the patches the bake exists to repair.
# Set RETOPO_BAKE_CHECK=0 to keep every bake unconditionally.
_BAKE_CHECK = os.environ.get("RETOPO_BAKE_CHECK", "1") not in ("0", "false", "False")
BAKE_CHECK_SAMPLES = 20_000
BAKE_CHECK_MARGIN = 0.97        # the bake must cut the mean error by at least 3%
BAKE_CHECK_TRIM_DEG = 60.0
# ...and for UVs outside the unit square by more than this (tiling materials, trim sheets): the
# rasterizer cannot address texels outside [0,1], and a tiling texture is shared by every surface
# that repeats it.
BAKE_UV_RANGE_TOL = 0.01


def _bake_ineligible(g: GLTF2, mat, plans: List[Dict]) -> Optional[str]:
    """Why a material's primitives cannot share one baked normal map, or None if they can."""
    if mat is None:
        return "no material to attach a normal map to"
    for plan in plans:
        if plan["attrs"].get("TEXCOORD_0") is None or plan["attrs_in"].get("TEXCOORD_0") is None:
            return "no TEXCOORD_0"
    nt = mat.normalTexture
    if nt is not None:
        if (nt.texCoord or 0) != 0:
            return f"normal map samples TEXCOORD_{nt.texCoord}"
        if "KHR_texture_transform" in (nt.extensions or {}):
            return "normal map uses KHR_texture_transform"
    lo, hi = -BAKE_UV_RANGE_TOL, 1.0 + BAKE_UV_RANGE_TOL
    for plan in plans:
        for uv in (plan["attrs_in"]["TEXCOORD_0"], plan["attrs"]["TEXCOORD_0"]):
            if len(uv) and (float(uv.min()) < lo or float(uv.max()) > hi):
                return "UVs outside 0-1 (tiling texture / trim sheet)"
    ov = _uv_overlap_frac([p["attrs_in"]["TEXCOORD_0"] for p in plans], [p["F"] for p in plans])
    if ov > BAKE_MAX_UV_OVERLAP:
        return f"overlapping UVs ({ov:.0%} of the layout is reused)"
    return None


def _bake_helps(checks: List[Dict], final_map: np.ndarray, src_map: Optional[np.ndarray],
                scale: float) -> Dict:
    """Mean shading error vs the source, baked vs not, over a sample of source vertices.

    `checks`: per primitive {"hi": high-poly dict, "hi_map", "plan", "n_lo", "t_lo"}. The unbaked
    variant is exactly what would be written without the bake: the carried NORMAL / TANGENT and the
    material's original map."""
    from .bake_normals import shade_at
    rng = np.random.default_rng(0)
    per = max(BAKE_CHECK_SAMPLES // max(len(checks), 1), 256)
    err_b, err_p = [], []
    fmap = final_map.astype(np.float32) / 255.0
    for c in checks:
        hi, plan = c["hi"], c["plan"]
        P_hi = np.asarray(hi["P"], dtype=np.float64)
        F_hi = np.asarray(hi["F"], dtype=np.int64)
        idx = np.unique(F_hi.reshape(-1))
        if len(idx) > per:
            idx = rng.choice(idx, per, replace=False)
        Q = P_hi[idx]
        N_hi = hi.get("NORMAL")
        if N_hi is None:
            from .bake_normals import _smooth_normals
            N_hi = _smooth_normals(P_hi, F_hi)
        ref = shade_at(Q, P_hi, F_hi, hi["UV"], N_hi, hi.get("TANGENT"), c["hi_map"], scale)
        pos = plan["pos"].astype(np.float64)
        uv = plan["attrs"]["TEXCOORD_0"].astype(np.float64)
        nb = shade_at(Q, pos, plan["faces"], uv, c["n_lo"], c["t_lo"], fmap, 1.0)
        n_carried = plan["attrs"].get("NORMAL")
        if n_carried is None:
            from .bake_normals import _smooth_normals
            n_carried = _smooth_normals(pos, plan["faces"])
        npl = shade_at(Q, pos, plan["faces"], uv, n_carried, plan["attrs"].get("TANGENT"),
                       src_map, scale)
        err_b.append(np.degrees(np.arccos(np.clip((nb * ref).sum(1), -1, 1))))
        err_p.append(np.degrees(np.arccos(np.clip((npl * ref).sum(1), -1, 1))))
    eb, ep = np.concatenate(err_b), np.concatenate(err_p)
    ok = (eb < BAKE_CHECK_TRIM_DEG) & (ep < BAKE_CHECK_TRIM_DEG)
    if ok.mean() >= 0.5:
        eb, ep = eb[ok], ep[ok]
    mb, mp = float(eb.mean()), float(ep.mean())
    return {"baked_mean_deg": round(mb, 3), "unbaked_mean_deg": round(mp, 3),
            "samples": int(len(eb)), "kept": mb < mp * BAKE_CHECK_MARGIN}


def _bake_stage(g: GLTF2, blob: bytes, plans: List[Dict],
                reference_glb: Optional[str] = None, res_hint: Optional[int] = None
                ) -> Tuple[Dict, List[Dict]]:
    """Bake the high-poly detail of every decimated primitive into a normal map, ONE MAP PER
    MATERIAL: primitives that share a material share its UV atlas, so their bakes are composited
    texel-by-texel into a single map. (Baking them one by one into the same slot, which this did,
    leaves the material holding whichever primitive was baked last.)

    Mutates each baked plan's NORMAL/TANGENT attributes in place — the bake is only valid in the
    tangent frame it was computed against, so the map and the basis must ship together. A material
    that cannot be baked safely (tiling or overlapping UVs, a transformed normal map, no UVs) is
    skipped and its primitives keep the attributes the decimator carried over from the source.

    Returns (stats, [{"material", "data", "mime", "prims"}]) — `_assign_baked_maps` wires them in.
    """
    from . import bake_normals as bake

    ref = _reference_highpoly(reference_glb) if reference_glb else {}
    groups: Dict[Optional[int], List[Dict]] = {}
    for plan in plans:
        groups.setdefault(plan["prim"].material, []).append(plan)

    baked_maps: List[Dict] = []
    details: List[Dict] = []
    updates: List[Tuple[Dict, np.ndarray, np.ndarray]] = []
    baked = 0
    for mat_idx, gp in groups.items():
        mat = (g.materials[mat_idx] if (mat_idx is not None and g.materials
                                        and mat_idx < len(g.materials)) else None)
        why = _bake_ineligible(g, mat, gp)
        if why:
            logger.info("retopotool: not baking material %s: %s", mat_idx, why)
            details.append({"material": mat_idx, "skipped": why})
            continue
        src_img_idx = (_texture_image_index(g, mat.normalTexture.index)
                       if mat.normalTexture is not None else None)
        src_raw = _image_bytes(g, blob, src_img_idx)
        src_map = bake.decode_image(src_raw) if src_raw is not None else None
        scale = float(mat.normalTexture.scale if (mat.normalTexture is not None
                                                  and mat.normalTexture.scale is not None) else 1.0)
        if res_hint:
            res = bake.pick_resolution([res_hint], allow_small=True)
        else:
            # The map ships at the size the material's own textures already are: the source normal
            # map, else the base colour (else BAKE_MIN_RES). A prop with a 512 texture does not
            # want a 2048 bake.
            size = _image_size(src_raw)
            if size is None and mat.pbrMetallicRoughness is not None \
                    and mat.pbrMetallicRoughness.baseColorTexture is not None:
                size = _image_size(_image_bytes(
                    g, blob, _texture_image_index(g, mat.pbrMetallicRoughness.baseColorTexture.index)))
            res = bake.pick_resolution([size or bake.BAKE_MIN_RES], allow_small=True)

        rgb_acc = np.zeros((res, res, 3), dtype=np.float32)
        cov_acc = np.zeros((res, res), dtype=bool)
        results = []
        checks = []
        gdet: List[Dict] = []
        for plan in gp:
            r = ref.get(plan["key"])
            if r is not None:
                hi, hi_map, hi_scale = r, r.get("normal_map"), scale
            else:
                hi = {"P": plan["P"].astype(np.float64), "F": plan["F"],
                      "UV": plan["attrs_in"]["TEXCOORD_0"].astype(np.float64),
                      "NORMAL": plan["attrs_in"].get("NORMAL"),
                      "TANGENT": plan["attrs_in"].get("TANGENT")}
                hi_map, hi_scale = src_map, scale
            rgb, cov, n_lo, t_lo, st = bake.bake_texels(
                hi["P"], hi["F"], hi["UV"], hi.get("NORMAL"), hi.get("TANGENT"), hi_map,
                plan["pos"].astype(np.float64), plan["faces"],
                plan["attrs"]["TEXCOORD_0"].astype(np.float64), res,
                N_lo_guide=plan["attrs"].get("NORMAL"), src_normal_scale=hi_scale)
            rgb_acc[cov] = rgb[cov]
            cov_acc |= cov
            results.append((plan, n_lo, t_lo))
            checks.append({"hi": hi, "hi_map": hi_map, "plan": plan, "n_lo": n_lo, "t_lo": t_lo})
            st["material"] = mat_idx
            gdet.append(st)
        final = bake.finish_map(rgb_acc, cov_acc)
        if _BAKE_CHECK:
            verdict = _bake_helps(checks, final, src_map, scale)
            if not verdict["kept"]:
                logger.info("retopotool: material %s keeps its original normals and map: %s",
                            mat_idx, verdict)
                details.append({"material": mat_idx, "check": verdict,
                                "skipped": "the bake did not improve shading "
                                           f"({verdict['baked_mean_deg']} vs "
                                           f"{verdict['unbaked_mean_deg']} deg mean error)"})
                continue
            for st in gdet:
                st["check"] = verdict
        details += gdet
        raw, mime = bake.encode_map(final, src_raw)
        updates += results
        baked_maps.append({"material": mat_idx, "data": raw, "mime": mime,
                           "prims": [p["prim"] for p in gp]})
    # Only now, with every material baked, touch the plans: an exception above must leave every
    # primitive with the attributes the decimator carried, not half of them re-based.
    for plan, n_lo, t_lo in updates:
        plan["attrs"]["NORMAL"] = n_lo
        plan["attrs"]["TANGENT"] = t_lo
        baked += 1
    stats = {
        "normal_map_baked": baked > 0,
        "bake_resolution": max([d.get("resolution", 0) for d in details], default=0) or None,
        "bake_skipped": [d for d in details if "skipped" in d],
        "bake": [d for d in details if "skipped" not in d],
    }
    return stats, baked_maps


def _texture_refs(g: GLTF2) -> Dict[int, int]:
    """How many texture-info slots across all materials (core AND extension slots, e.g.
    KHR_materials_clearcoat's normal texture) point at each texture index."""
    counts: Dict[int, int] = {}

    def walk(o, under_texture=False):
        if o is None:
            return
        if isinstance(o, dict):
            if under_texture and isinstance(o.get("index"), int):
                counts[o["index"]] = counts.get(o["index"], 0) + 1
            for k, v in o.items():
                walk(v, isinstance(k, str) and k.lower().endswith("texture"))
            return
        if isinstance(o, (list, tuple)):
            for v in o:
                walk(v, under_texture)
            return
        if hasattr(o, "__dict__"):
            name = type(o).__name__
            if (name.endswith("TextureInfo") or name.endswith("MaterialTexture")) \
                    and isinstance(getattr(o, "index", None), int):
                counts[o.index] = counts.get(o.index, 0) + 1
            for k, v in vars(o).items():
                if k == "index":
                    continue
                walk(v, isinstance(k, str) and k.lower().endswith("texture"))

    for mat in (g.materials or []):
        walk(mat)
    return counts


def _assign_baked_maps(g: GLTF2, baked_maps: List[Dict]
                       ) -> Tuple[Dict[int, Tuple[bytes, str]], List[Tuple[int, bytes, str]]]:
    """Decide where each baked map goes WITHOUT changing anything outside the baked primitives.

    The map replaces the source normal image in place only when nothing else can see that image:
    one texture uses it, one material slot uses that texture (the baked material's normal slot) and
    every primitive using the material was baked into this map. Otherwise the material — cloned
    first if primitives outside this bake also use it — gets a freshly minted image. A tiling
    normal map shared by forty wall pieces stays exactly what it was for the other thirty-nine.

    Returns ({image index: (bytes, mime)} to overwrite, [(material index, bytes, mime)] needing a
    new image).
    Mutates `g.materials` (clones, normalTexture) and the baked primitives' `material`.
    """
    import copy

    overrides: Dict[int, Tuple[bytes, str]] = {}
    additions: List[Tuple[int, bytes, str]] = []
    tex_refs = _texture_refs(g)
    img_tex: Dict[int, int] = {}
    for ti in range(len(g.textures or [])):
        ii = _texture_image_index(g, ti)
        if ii is not None:
            img_tex[ii] = img_tex.get(ii, 0) + 1
    users: Dict[int, set] = {}
    for mesh in (g.meshes or []):
        for prim in mesh.primitives:
            if prim.material is not None:
                users.setdefault(prim.material, set()).add(id(prim))

    for entry in baked_maps:
        m = entry["material"]
        group = {id(p) for p in entry["prims"]}
        if not users.get(m, set()) <= group:
            clone = copy.deepcopy(g.materials[m])
            clone.name = f"{g.materials[m].name or 'material'}_baked"
            g.materials.append(clone)
            m_new = len(g.materials) - 1
            for p in entry["prims"]:
                p.material = m_new
            users[m] = users[m] - group
            users[m_new] = set(group)
            m = m_new
            exclusive = False            # the source image is still in use by the original
        else:
            exclusive = True
        mat = g.materials[m]
        nt = mat.normalTexture
        src_img = _texture_image_index(g, nt.index) if nt is not None else None
        if (exclusive and src_img is not None and src_img not in overrides
                and img_tex.get(src_img, 0) == 1 and tex_refs.get(nt.index, 0) == 1):
            overrides[src_img] = (entry["data"], entry["mime"])
            # the baked map is in TEXCOORD_0 space, untransformed, at unit strength
            nt.texCoord = 0
            nt.scale = 1.0
            if nt.extensions:
                nt.extensions.pop("KHR_texture_transform", None)
        else:
            additions.append((m, entry["data"], entry["mime"]))
    return overrides, additions


# Skin attributes ride the decimation separately from _ATTR_NAMES: they must never go through the
# "integer attribute -> float 0..1" normalization (JOINTS_0 holds bone INDICES) and must not be
# written back as FLOAT. See `_read_skin_attrs` / `_write_skin_attrs`.
_SKIN_NAMES = ("JOINTS_0", "WEIGHTS_0", "JOINTS_1", "WEIGHTS_1")


def _read_skin_attrs(g: GLTF2, blob: bytes, at) -> Dict[str, np.ndarray]:
    """Read JOINTS_*/WEIGHTS_* without the dtype-driven rescale the shading attributes get.

    `JOINTS_0` is UNSIGNED_BYTE with `normalized=false` — those are bone indices, and the generic
    reader's `arr / iinfo(dtype).max` would turn joint 66 into 0.259 and silently destroy the rig.
    `WEIGHTS_0` may legitimately be a normalized ubyte/ushort, so it is rescaled ONLY when the
    accessor actually says `normalized` (the generic reader keys off dtype, which is not the same
    question).
    """
    out: Dict[str, np.ndarray] = {}
    for name in _SKIN_NAMES:
        idx = getattr(at, name, None)
        if idx is None:
            continue
        arr = _acc(g, blob, idx)
        if name.startswith("WEIGHTS") and np.issubdtype(arr.dtype, np.integer):
            if getattr(g.accessors[idx], "normalized", False):
                arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
            else:
                raise ValueError(f"{name} is a non-normalized integer accessor — refusing to guess "
                                 "its scale")
        out[name] = arr
    return out


def _resolve_boosts(profile: str, has_skin: bool, lod: bool,
                    head_boost: Optional[float], hand_boost: Optional[float]
                    ) -> Tuple[float, float]:
    """Head/hand importance boosts for a decimation.

    `profile`: "prop" = uniform budget (the warp is the identity), "character" = the humanoid
    warp, "auto" = character when the file carries a skin, prop otherwise. The warp assumes a
    Y-up humanoid in an A/T-pose; on a barrel or a wall section it just magnifies whatever happens
    to be in the top 16% or at the X extremes, so it is never applied unless asked for.
    Explicit `head_boost` / `hand_boost` override the profile.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
    character = profile == "character" or (profile == "auto" and has_skin)
    if character:
        dh, dn = (_LOD_HEAD_BOOST, _LOD_HAND_BOOST) if lod else (_HEAD_BOOST, _HAND_BOOST)
    else:
        dh = dn = 1.0
    return (dh if head_boost is None else float(head_boost),
            dn if hand_boost is None else float(hand_boost))


def _resolve_seam_lock(profile: str, has_skin: bool) -> bool:
    """Seam fins (`_seam_fins`) for props, not for characters — unless RETOPO_SEAM_LOCK says
    otherwise. A prop's seams sit on flat faces where a collapse across them is free and visible;
    the character pipeline was measured and tuned without them (its seams sit on curved skin, and
    locking ~40% of a generated character's vertices to seams costs budget where it is needed)."""
    env = os.environ.get("RETOPO_SEAM_LOCK")
    if env is not None:
        return env not in ("0", "false", "False")
    return not (profile == "character" or (profile == "auto" and has_skin))


def decimate_source_glb(source_glb_path: str, output_glb_path: str, target_ratio: float = 0.5, *,
                        bake: bool = True,
                        profile: str = "prop",
                        head_boost: Optional[float] = None,
                        hand_boost: Optional[float] = None) -> Dict:
    """Write a seam-preserving decimated copy of an UNRIGGED GLB — a prop, an environment piece or
    a character before rigging. `target_ratio` = fraction of triangles to KEEP.
    Textures/materials/nodes preserved; buffer repacked. Returns stats. Raises ValueError for
    skinned/animated/morphed inputs (optimize runs before rigging).

    `bake=False` skips the normal-map bake (plain decimation). `profile` picks the density
    allocation: "prop" (default, uniform) or "character" (head and hands keep more of the budget);
    `head_boost` / `hand_boost` override it.

    For the rigged LOD ladder use `decimate_rigged_glb`, which keeps this function's
    "optimize before you rig" contract intact instead of loosening it.
    """
    return _decimate_glb(source_glb_path, output_glb_path, target_ratio=target_ratio, bake=bake,
                         profile=profile, head_boost=head_boost, hand_boost=hand_boost)


def _assert_skin_intact(path: str) -> None:
    """Re-open the written LOD and check the invariants a broken rig violates SILENTLY.

    None of these raise on their own: a stale inverse-bind index, a joint index scaled to 0..1, or a
    weight row that no longer sums to 1 all produce a file that loads and then deforms into garbage.
    Checking the file we just wrote (rather than the arrays we think we wrote) also catches an
    accessor mis-wire in the repack itself.
    """
    g = GLTF2().load(path)
    blob = g.binary_blob()
    for skin in (g.skins or []):
        njoints = len(skin.joints or [])
        if skin.inverseBindMatrices is not None:
            ibm = g.accessors[skin.inverseBindMatrices]
            if ibm.type != "MAT4" or ibm.componentType != FLOAT:
                raise ValueError(f"inverse-bind accessor is {ibm.type}/{ibm.componentType}, "
                                 "expected MAT4/FLOAT — the repack cross-wired it")
            if ibm.count != njoints:
                raise ValueError(f"inverse-bind count {ibm.count} != {njoints} joints")
        for mesh in (g.meshes or []):
            for prim in mesh.primitives:
                at = prim.attributes
                if at.POSITION is None or at.JOINTS_0 is None:
                    continue
                nv = g.accessors[at.POSITION].count
                ja = g.accessors[at.JOINTS_0]
                if ja.componentType not in (UNSIGNED_BYTE, UNSIGNED_SHORT):
                    raise ValueError("JOINTS_0 must be an unsigned integer accessor — a float one "
                                     "means the bone indices were rescaled to 0..1")
                if getattr(ja, "normalized", False):
                    raise ValueError("JOINTS_0 is flagged normalized; bone indices are not a ratio")
                j = _acc(g, blob, at.JOINTS_0)
                if int(j.max()) >= njoints:
                    raise ValueError(f"joint index {int(j.max())} out of range for {njoints} joints")
                w = _acc(g, blob, at.WEIGHTS_0).astype(np.float64)
                if len(j) != nv or len(w) != nv:
                    raise ValueError("skin attribute count != vertex count")
                if not np.allclose(w.sum(1), 1.0, atol=1e-3):
                    raise ValueError("skin weights do not sum to 1")
                idx = _acc(g, blob, prim.indices) if prim.indices is not None else None
                if idx is not None and int(idx.max()) >= nv:
                    raise ValueError("index buffer references a vertex that does not exist")


def decimate_rigged_glb(source_glb_path: str, output_glb_path: str, target_triangles: int,
                        texture_size: Optional[int] = None, bake: bool = True,
                        bake_reference: Optional[str] = None,
                        head_boost: Optional[float] = None,
                        hand_boost: Optional[float] = None,
                        profile: str = "auto") -> Dict:
    """Write one LOD of a GLB — rigged or not — keeping any skin intact.

    The same seam-preserving collapse as the unrigged path. Skin is resolved PER OUTPUT WELDED
    VERTEX and blended over the collapse cluster (see `_blend_skin_over_clusters` and the note at
    the end of `_decimate_primitive`) — NOT copied per UV wedge. Copying per wedge is what this did
    until 2026-09-05, and it gave the two sides of every UV seam different bones: 60% of seam weld
    groups at 30k, opening a measured 5 mm (p95) / 15 mm (p99) gap once the character was posed.
    Every static check passed throughout, because the split copies are duplicate vertices rather
    than neighbours across an edge.

    `profile="auto"` uses the humanoid head/hand warp for a skinned file and a uniform budget
    otherwise. The character boosts default LOWER here than for the 400k Optimize step: the warp's
    cost scales with how scarce the budget is, and at LOD resolutions the head boost was buying
    ~0.03 mm of head p95 with ~0.31 mm of body p95.

    `bake_reference` is the mesh to bake the normal map FROM, matched primitive by primitive (same
    mesh and primitive index); pass the full-resolution source so each rung of the ladder is baked
    against the best available surface rather than against the rung above it. `texture_size`
    downscales the textures and sets the bake resolution.

    The skeleton is untouched: `skins[0].joints`, the node hierarchy and the inverse-bind data all
    ride through unchanged (the IBM ACCESSOR INDEX is re-pointed, which the repack must do and is
    the single most destructive thing to get wrong here).
    """
    return _decimate_glb(source_glb_path, output_glb_path, lod=True, profile=profile,
                         head_boost=head_boost, hand_boost=hand_boost,
                         target_triangles=int(target_triangles), allow_skin=True,
                         texture_size=texture_size, bake=bake, bake_reference=bake_reference)


# Every attribute the decimator treats as SHADING data: read as float (normalized integers
# rescaled), re-derived where it has to be (UVs resampled, NORMAL/TANGENT re-baked) and written
# back as FLOAT. Any other attribute (TEXCOORD_n beyond these, COLOR_1, `_CUSTOM`/`_FEATURE_ID_0`)
# is carried as an exact per-vertex copy in its original component type.
def _is_shading_attr(name: str) -> bool:
    return name in ("NORMAL", "TANGENT") or name.startswith("TEXCOORD_") or name.startswith("COLOR_")


def _dequantize(g: GLTF2, idx: int, arr: np.ndarray, force: bool = False) -> np.ndarray:
    """Integer accessor -> float32, honouring `normalized` (KHR_mesh_quantization): a normalized
    signed value maps to max(c / max, -1), an unsigned one to c / max, a non-normalized one is
    taken at face value."""
    if not np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32)
    if force or getattr(g.accessors[idx], "normalized", False):
        out = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        return np.maximum(out, -1.0) if np.issubdtype(arr.dtype, np.signedinteger) else out
    return arr.astype(np.float32)


def _read_position(g: GLTF2, blob: bytes, idx: int) -> np.ndarray:
    return _dequantize(g, idx, _acc(g, blob, idx))


def _read_float_attr(g: GLTF2, blob: bytes, name: str, idx: int) -> np.ndarray:
    # COLOR_n integers are normalized by definition, flag or not
    return _dequantize(g, idx, _acc(g, blob, idx), force=name.startswith("COLOR_"))


def _prim_attr_names(prim) -> List[str]:
    return [k for k, v in vars(prim.attributes).items() if v is not None and k != "POSITION"]


def _decimate_mesh_jointly(parts: List[Dict], target_ratio: float, stats: Dict,
                           head_boost: Optional[float], hand_boost: Optional[float],
                           seam_lock: bool = False) -> List[Optional[Dict]]:
    """Decimate every triangle primitive of ONE mesh together, then split the result back.

    A multi-material mesh is one surface cut into primitives along material borders. Decimated one
    primitive at a time, each side of a border collapses on its own schedule and the border opens
    into a crack (measured on a two-material crate: 55 boundary edges at 10%). Concatenated, the
    weld joins the two sides of every border exactly like a UV seam, the collapse moves them in
    lockstep, and each output face goes back to the primitive its ancestor face came from — raw
    vertices never cross primitives, so neither can the island-constrained wedge re-split.

    `parts`: [{"P", "F", "attrs"}] per primitive. Returns, per part, {"pos", "faces", "attrs"} or
    None where that primitive lost every face (the caller keeps the original) — or None for the
    whole mesh when it is too small to decimate.
    """
    sizes = [len(p["P"]) for p in parts]
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    P = np.concatenate([p["P"] for p in parts]).astype(np.float32)
    F = np.concatenate([p["F"] + offs[i] for i, p in enumerate(parts)])
    owner = np.repeat(np.arange(len(parts)), sizes)

    # Union of attribute names; a part without one is zero-filled (and never written back), widths
    # are padded to the widest (COLOR_0 VEC3 + VEC4 -> VEC4 with alpha 1) and sliced back on output.
    names: List[str] = []
    for p in parts:
        names += [n for n in p["attrs"] if n not in names]
    attrs: Dict[str, np.ndarray] = {}
    for n in names:
        arrs = [p["attrs"].get(n) for p in parts]
        width = max(1 if a.ndim == 1 else a.shape[1] for a in arrs if a is not None)
        dtype = np.result_type(*[a.dtype for a in arrs if a is not None])
        cols = []
        for a, sz in zip(arrs, sizes):
            if a is None:
                cols.append(np.zeros((sz, width), dtype=dtype))
                continue
            a2 = a.reshape(sz, -1).astype(dtype, copy=False)
            if a2.shape[1] < width:
                pad = np.zeros((sz, width - a2.shape[1]), dtype=dtype)
                if n.startswith("COLOR_"):
                    pad[:] = 1
                a2 = np.concatenate([a2, pad], axis=1)
            cols.append(a2)
        cat = np.concatenate(cols)
        attrs[n] = cat.reshape(-1) if width == 1 and all(
            a is None or a.ndim == 1 for a in arrs) else cat

    res = _decimate_primitive(P, F, attrs, target_ratio, stats=stats, head_boost=head_boost,
                              hand_boost=hand_boost, return_source=True, seam_lock=seam_lock,
                              labels=owner if len(parts) > 1 else None)
    if res is None:
        return [None] * len(parts)
    pos, faces, a_out, vr = res
    v_owner = owner[vr]
    f_owner = v_owner[faces[:, 0]]
    out: List[Optional[Dict]] = []
    for i, p in enumerate(parts):
        fi = faces[f_owner == i]
        if len(fi) == 0:
            out.append(None)
            continue
        used, inv = np.unique(fi.reshape(-1), return_inverse=True)
        pa = {}
        for n, a in p["attrs"].items():
            src = a_out[n][used].reshape(len(used), -1)
            src = src[:, 0] if a.ndim == 1 else src[:, :a.shape[1]]
            if not _is_shading_attr(n) and n not in _SKIN_NAMES:
                src = src.astype(a.dtype)          # custom attributes keep their exact values
            pa[n] = src
        out.append({"pos": pos[used], "faces": inv.reshape(-1, 3).astype(np.int64), "attrs": pa})
    return out


def _decimate_glb(source_glb_path: str, output_glb_path: str, *,
                  target_ratio: float = 0.5,
                  target_triangles: Optional[int] = None,
                  allow_skin: bool = False,
                  lod: bool = False,
                  profile: str = "prop",
                  texture_size: Optional[int] = None,
                  bake: bool = True,
                  bake_reference: Optional[str] = None,
                  head_boost: Optional[float] = None,
                  hand_boost: Optional[float] = None) -> Dict:
    """Shared implementation. `allow_skin` opts into carrying JOINTS_*/WEIGHTS_* and remapping the
    skin's inverse-bind accessor; everything else is identical for both entry points."""
    if not os.path.exists(source_glb_path):
        raise FileNotFoundError(source_glb_path)
    # pygltflib dispatches save/load on the FILE EXTENSION: anything but `.glb` is written as a
    # JSON glTF plus a `.bin` sidecar, which loads back with no binary blob and no error. That is
    # exactly how an atomic-publish temp name like `foo.glb.tmp` silently produced a non-GLB.
    # Checked up front so a bad path costs nothing.
    if not output_glb_path.lower().endswith(".glb"):
        raise ValueError(f"output must end in .glb (pygltflib writes anything else as JSON+bin): "
                         f"{output_glb_path}")

    g = GLTF2().load(source_glb_path)
    blob = g.binary_blob()
    if blob is None:
        raise ValueError(f"{source_glb_path} has no binary chunk — only self-contained .glb "
                         "files are supported")
    orig_size = os.path.getsize(source_glb_path)
    if g.animations:
        raise ValueError("mesh is animated — the repack does not re-index animation samplers")
    if g.skins and not allow_skin:
        raise ValueError("mesh is rigged/animated — optimize it before rigging, not after")
    if allow_skin and len(g.skins or []) > 1:
        raise ValueError(f"expected a single skin, found {len(g.skins)} — cannot LOD safely")
    # 8-INFLUENCE RIGS ARE REFUSED, NOT SILENTLY MANGLED. The writer normalizes each WEIGHTS_n row
    # to 1 independently, so a rig whose WEIGHTS_0 + WEIGHTS_1 sum to 1 together would be written
    # summing to 2 — and `_assert_skin_intact` only inspects set 0, so nothing downstream would
    # notice. The blend is defined for the 4-influence pair only. Supporting set 1 would mean
    # merging both sets and re-splitting them together.
    if allow_skin:
        for mesh in (g.meshes or []):
            for prim in mesh.primitives:
                if getattr(prim.attributes, "JOINTS_1", None) is not None:
                    raise ValueError("mesh has 8 skin influences (JOINTS_1); this path handles 4 "
                                     "— merging the two weight sets is not implemented")
    head_boost, hand_boost = _resolve_boosts(profile, bool(g.skins), lod, head_boost, hand_boost)
    seam_lock = _resolve_seam_lock(profile, bool(g.skins))

    def _is_tri(prim) -> bool:
        if prim.attributes.POSITION is None or (prim.mode if prim.mode is not None else 4) != 4:
            return False
        n = (g.accessors[prim.indices].count if prim.indices is not None
             else g.accessors[prim.attributes.POSITION].count)
        return n % 3 == 0 and n > 0

    if target_triangles is not None:
        total = sum((g.accessors[p.indices].count if p.indices is not None
                     else g.accessors[p.attributes.POSITION].count) // 3
                    for m in (g.meshes or []) for p in m.primitives if _is_tri(p))
        if total <= 0:
            raise ValueError("nothing to decimate (no triangle geometry found)")
        target_ratio = target_triangles / float(total)
    target_ratio = float(np.clip(target_ratio, 0.02, 0.95))
    # Fail LOUD on GLB features the repack below cannot re-index (better a clear error than a
    # silently corrupted file): sparse accessors, compressed/instanced extensions, extra buffers.
    if len(g.buffers or []) > 1:
        raise ValueError("multi-buffer GLB — cannot optimize safely")
    unsupported_exts = {"KHR_draco_mesh_compression", "EXT_meshopt_compression",
                        "KHR_meshopt_compression", "EXT_mesh_gpu_instancing"} & set(
        (g.extensionsUsed or []) + (g.extensionsRequired or []))
    if unsupported_exts:
        raise ValueError(f"unsupported GLB extensions: {sorted(unsupported_exts)} — decompress "
                         "the file first (e.g. `gltf-transform copy in.glb out.glb`)")
    for a in (g.accessors or []):
        if getattr(a, "sparse", None):
            raise ValueError("sparse accessors — cannot optimize safely")
    for mesh in (g.meshes or []):
        for prim in mesh.primitives:
            if prim.targets:
                raise ValueError("mesh has morph targets — cannot optimize safely")

    tri_before = tri_after = vert_before = vert_after = 0
    plans: List[Dict] = []
    density: List[Dict] = []
    total_verts = 0
    for mi, mesh in enumerate(g.meshes or []):
        parts: List[Dict] = []
        for pi, prim in enumerate(mesh.primitives):
            at = prim.attributes
            if at.POSITION is None:
                continue
            n_pos = g.accessors[at.POSITION].count
            total_verts += n_pos
            if total_verts > MAX_TOTAL_VERTS:
                raise ValueError(f"mesh too large to optimize: >{MAX_TOTAL_VERTS} vertices")
            vert_before += n_pos
            # only plain indexed/soup TRIANGLES (mode 4) decimate; strips/fans/lines/points and
            # non-multiple-of-3 counts pass through verbatim
            if not _is_tri(prim):
                vert_after += n_pos
                continue
            P = _read_position(g, blob, at.POSITION)
            F = (_acc(g, blob, prim.indices).astype(np.int64).reshape(-1, 3)
                 if prim.indices is not None else np.arange(len(P), dtype=np.int64).reshape(-1, 3))
            tri_before += len(F)
            if F.max() >= len(P) or F.min() < 0:
                raise ValueError(f"mesh {mi} primitive {pi}: index out of range")
            # Only the vertices the faces use. Several exporters give every primitive of a mesh
            # the SAME vertex arrays and a different index buffer; carried whole, each primitive
            # would drag every other primitive's vertices into its weld.
            used, F_local = np.unique(F.reshape(-1), return_inverse=True)
            F_local = F_local.reshape(-1, 3).astype(np.int64)
            attrs: Dict[str, np.ndarray] = {}
            meta: Dict[str, Tuple[int, str, bool]] = {}
            for name in _prim_attr_names(prim):
                a_idx = getattr(at, name)
                if name in _SKIN_NAMES:
                    continue
                if _is_shading_attr(name):
                    attrs[name] = _read_float_attr(g, blob, name, a_idx)[used]
                else:
                    acc = g.accessors[a_idx]
                    attrs[name] = _acc(g, blob, a_idx)[used]
                    meta[name] = (acc.componentType, acc.type, bool(acc.normalized))
            # Skin rides in the SAME attrs dict so it goes through the same exact-copy gather
            # (`arr[vr]`) as the shading attributes — never nearest-neighbour transferred.
            skin_in = ({k: v[used] for k, v in _read_skin_attrs(g, blob, at).items()}
                       if allow_skin else {})
            attrs.update(skin_in)
            parts.append({"pi": pi, "prim": prim, "P": P[used], "F": F_local, "attrs": attrs,
                          "meta": meta, "n_pos": n_pos})
        if not parts:
            continue
        mesh_stats_d: Dict = {}
        results = _decimate_mesh_jointly(parts, target_ratio, mesh_stats_d, head_boost, hand_boost,
                                         seam_lock=seam_lock)
        if any(r is not None for r in results):
            density.append(mesh_stats_d)
        for part, r in zip(parts, results):
            if r is None:
                vert_after += part["n_pos"]
                tri_after += len(part["F"])
                continue
            a_out = dict(r["attrs"])
            skin_out = {k: a_out.pop(k) for k in list(a_out) if k in _SKIN_NAMES}
            vert_after += len(r["pos"])
            tri_after += len(r["faces"])
            plans.append({"prim": part["prim"], "key": (mi, part["pi"]), "pos": r["pos"],
                          "faces": r["faces"], "attrs": a_out, "skin": skin_out,
                          "meta": part["meta"], "P": part["P"], "F": part["F"],
                          "attrs_in": {k: v for k, v in part["attrs"].items()
                                       if k not in _SKIN_NAMES}})

    if not plans:
        raise NothingToDecimate("nothing to decimate (no triangle primitive is large enough — "
                         "at least 64 faces — to reduce)")

    # ---- bake the removed relief into the normal map, before anything is written ----
    # Gated so a failure here degrades to plain decimation rather than losing the optimize
    # entirely; RETOPO_OPT_BAKE_NORMALS=0 turns it off outright.
    bake_stats: Dict = {"normal_map_baked": False, "bake_resolution": None}
    baked_maps: List[Dict] = []
    if bake and os.environ.get("RETOPO_OPT_BAKE_NORMALS", "1") not in ("0", "false", "False"):
        try:
            bake_stats, baked_maps = _bake_stage(g, blob, plans, reference_glb=bake_reference,
                                                 res_hint=texture_size)
        except Exception:
            logger.exception("retopotool: normal-map bake failed; writing the decimated mesh "
                             "with its original shading attributes")
            bake_stats = {"normal_map_baked": False, "bake_resolution": None,
                          "bake_error": "bake failed — see logs"}
            baked_maps = []
    image_overrides, image_additions = _assign_baked_maps(g, baked_maps)
    for plan in plans:      # the high-poly arrays are only needed by the bake; free them now
        plan.pop("P", None)
        plan.pop("F", None)
        plan.pop("attrs_in", None)

    # ---- repack: images verbatim + kept accessors + new decimated geometry ----
    new_blob = bytearray()
    new_views: List[BufferView] = []
    new_accessors: List[Accessor] = []

    def add_view(raw: bytes, target: Optional[int] = None) -> int:
        while len(new_blob) % 4:
            new_blob.append(0)
        off = len(new_blob)
        new_blob.extend(raw)
        new_views.append(BufferView(buffer=0, byteOffset=off, byteLength=len(raw), target=target))
        return len(new_views) - 1

    def add_accessor(arr: np.ndarray, comp: int, gtype: str, minmax: bool = False,
                     normalized: bool = False) -> int:
        data = np.ascontiguousarray(arr, dtype=_CT[comp])
        bv = add_view(data.tobytes())
        a = Accessor(bufferView=bv, byteOffset=0, componentType=comp, count=len(data), type=gtype,
                     normalized=bool(normalized) or None)
        if minmax:
            a.min = data.min(0).tolist() if data.ndim > 1 else [data.min().item()]
            a.max = data.max(0).tolist() if data.ndim > 1 else [data.max().item()]
        new_accessors.append(a)
        return len(new_accessors) - 1

    img_view: Dict[int, int] = {}
    img_mime: Dict[int, str] = {}
    for i, img in enumerate(g.images or []):
        if i in image_overrides:                      # baked normal map replaces the source one
            data, mime = image_overrides[i]
            img_view[i] = add_view(data)
            img_mime[i] = mime
            continue
        if img.bufferView is None:
            continue
        bv = g.bufferViews[img.bufferView]
        raw = bytes(blob[(bv.byteOffset or 0):(bv.byteOffset or 0) + bv.byteLength])
        if texture_size:
            # Pillow LANCZOS, JPEG q90 / PNG. A baked normal map is NOT resampled here — it comes
            # out of the bake already at the target resolution.
            mime = img.mimeType or ("image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png")
            try:
                smaller = _resize_image(raw, mime, int(texture_size))
            except Exception as exc:
                logger.warning("retopotool: texture resize failed (%s); keeping the original", exc)
                smaller = None
            if smaller is not None:
                raw = smaller
                # A WebP source re-encodes to PNG, so the texture must be rewired off
                # EXT_texture_webp below or a WebP-less reader errors on PNG bytes.
                img_mime[i] = "image/jpeg" if ("jpeg" in mime or "jpg" in mime) else "image/png"
        img_view[i] = add_view(raw)

    # Copy only accessors something still POINTS AT: the primitives that were NOT replaced (all of
    # their attributes, whatever their names) and the skin's inverse-bind matrices. Rig writers
    # that append to the buffer instead of repacking leave the pre-rig POSITION/NORMAL/TANGENT
    # behind — 9.0 MB, 24.8% of a 38 MB rig — referenced by nothing; dropping them is most of the
    # reason an LOD is small. An accessor a replaced primitive SHARES with a kept one (a line
    # primitive indexing the same vertex array, a primitive too small to decimate) stays.
    decimated_prims = {id(plan["prim"]) for plan in plans}
    referenced: set = set()
    for mesh in (g.meshes or []):
        for prim in mesh.primitives:
            if id(prim) in decimated_prims:
                continue
            referenced.update(v for k, v in vars(prim.attributes).items() if v is not None)
            if prim.indices is not None:
                referenced.add(prim.indices)
    for skin in (g.skins or []):
        if skin.inverseBindMatrices is not None:
            referenced.add(skin.inverseBindMatrices)

    acc_map: Dict[int, int] = {}
    dropped_orphans = 0
    for old_idx, a in enumerate(g.accessors or []):
        if old_idx not in referenced:
            dropped_orphans += 1
            continue
        if a.bufferView is None:
            # all-zeros accessor: no data to move, keep it as declared
            new_accessors.append(Accessor(componentType=a.componentType, count=a.count,
                                          type=a.type, normalized=a.normalized, min=a.min,
                                          max=a.max))
            acc_map[old_idx] = len(new_accessors) - 1
            continue
        arr = _acc(g, blob, old_idx)
        acc_map[old_idx] = add_accessor(arr, a.componentType, a.type, minmax=a.min is not None,
                                        normalized=bool(a.normalized))
    if dropped_orphans:
        logger.info("retopotool: dropped %d unreferenced or replaced accessor(s)", dropped_orphans)

    for mesh in (g.meshes or []):
        for prim in mesh.primitives:
            if id(prim) in decimated_prims:
                continue
            at = prim.attributes
            for name, v in list(vars(at).items()):
                if v is not None and v in acc_map:
                    setattr(at, name, acc_map[v])
            if prim.indices is not None and prim.indices in acc_map:
                prim.indices = acc_map[prim.indices]
    for i, img in enumerate(g.images or []):
        if i in img_view:
            img.bufferView = img_view[i]
            if i in img_mime:
                img.mimeType = img_mime[i]
                img.uri = None
                if img_mime[i] == "image/webp":
                    continue
                # An EXT_texture_webp texture now points at PNG bytes; rewire it to the core slot
                # or every reader that honours the extension decodes a PNG as WebP.
                for tex in (g.textures or []):
                    ext = (tex.extensions or {}).get("EXT_texture_webp") or {}
                    if ext.get("source") == i:
                        tex.source = i
                        tex.extensions.pop("EXT_texture_webp", None)

    # Materials that need a fresh map (no normal map before, or one that other materials share).
    for mat_idx, data, mime in image_additions:
        bv = add_view(data)
        g.images = list(g.images or [])
        g.images.append(Image(bufferView=bv, mimeType=mime, name=f"baked_normal_{mat_idx}"))
        g.textures = list(g.textures or [])
        old = g.materials[mat_idx].normalTexture
        sampler = (g.textures[old.index].sampler
                   if old is not None and old.index is not None and old.index < len(g.textures)
                   else None)
        if mime == "image/webp":
            # WebP only ever comes back for a source map that was WebP, so the file already
            # declares the extension; reference it the way that extension requires.
            g.textures.append(Texture(sampler=sampler, extensions={
                "EXT_texture_webp": {"source": len(g.images) - 1}}))
            g.extensionsUsed = sorted(set((g.extensionsUsed or []) + ["EXT_texture_webp"]))
        else:
            g.textures.append(Texture(source=len(g.images) - 1, sampler=sampler))
        g.materials[mat_idx].normalTexture = NormalMaterialTexture(
            index=len(g.textures) - 1, scale=1.0, texCoord=0)

    # Drop EXT_texture_webp from the declaration once no texture uses it any more — leaving it in
    # extensionsRequired forces every reader to support a codec the file no longer contains.
    if not any((tex.extensions or {}).get("EXT_texture_webp") for tex in (g.textures or [])):
        for lst in ("extensionsUsed", "extensionsRequired"):
            cur = getattr(g, lst, None)
            if cur and "EXT_texture_webp" in cur:
                setattr(g, lst, [e for e in cur if e != "EXT_texture_webp"])

    _GT = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}
    for plan in plans:
        prim = plan["prim"]
        meta = plan.get("meta", {})
        # Every attribute slot is rewritten from the plan; anything the plan does not carry would be
        # left holding an index into the OLD accessor table.
        for name in list(vars(prim.attributes)):
            if name != "POSITION":
                setattr(prim.attributes, name, None)
        prim.attributes.POSITION = add_accessor(plan["pos"], FLOAT, "VEC3", minmax=True)
        for name, arr in plan["attrs"].items():
            if name in meta:                          # custom attribute: original encoding
                comp, gtype, normd = meta[name]
                setattr(prim.attributes, name, add_accessor(arr, comp, gtype, normalized=normd))
                continue
            width = 1 if arr.ndim == 1 else arr.shape[1]
            setattr(prim.attributes, name,
                    add_accessor(arr.astype(np.float32), FLOAT, _GT[width]))
        # Skin: written with the RIGHT component types, not the FLOAT every other attribute gets.
        # glTF allows JOINTS_n only as UNSIGNED_BYTE/UNSIGNED_SHORT; a float VEC4 there is invalid
        # and three.js will not bind it.
        for name, arr in (plan.get("skin") or {}).items():
            if name.startswith("JOINTS"):
                j = np.rint(arr).astype(np.int64)
                if j.min() < 0:
                    raise ValueError(f"{name} has a negative joint index")
                comp = UNSIGNED_BYTE if int(j.max()) < 256 else UNSIGNED_SHORT
                setattr(prim.attributes, name, add_accessor(j, comp, "VEC4", normalized=False))
            else:
                w = np.asarray(arr, dtype=np.float32)
                # Defensive only: the gather copies a whole original row, so the sum is already
                # whatever the source had. A row that sums to 0 would leave its vertex unskinned.
                tot = w.sum(1, keepdims=True)
                dead = (tot < 1e-8).ravel()
                if dead.any():
                    raise ValueError(f"{int(dead.sum())} vertices ended with zero skin weight")
                setattr(prim.attributes, name, add_accessor(w / tot, FLOAT, "VEC4"))
        n_verts = len(plan["pos"])
        icomp = UNSIGNED_SHORT if n_verts <= 65535 else UNSIGNED_INT
        prim.indices = add_accessor(plan["faces"].reshape(-1), icomp, "SCALAR")
        prim.mode = None                                  # TRIANGLES, the default

    # THE trap of this whole path: the repack rebuilds `g.accessors` from scratch, so every index
    # held OUTSIDE a primitive has to be re-pointed. `skin.inverseBindMatrices` is the only one, and
    # nothing here touched it before — a stale index means the rig loads against the wrong (or an
    # out-of-range) bind pose, which shows up as an exploded mesh rather than an error.
    for skin in (g.skins or []):
        if skin.inverseBindMatrices is not None:
            if skin.inverseBindMatrices not in acc_map:
                raise ValueError("skin.inverseBindMatrices was retired by the repack — refusing to "
                                 "write a rig with no bind pose")
            skin.inverseBindMatrices = acc_map[skin.inverseBindMatrices]
            ibm = new_accessors[skin.inverseBindMatrices]
            if ibm.count != len(skin.joints):
                raise ValueError(f"inverse-bind count {ibm.count} != joint count {len(skin.joints)}")

    g.bufferViews = new_views
    g.accessors = new_accessors
    g.buffers = [Buffer(byteLength=len(new_blob))]
    g.set_binary_blob(bytes(new_blob))
    os.makedirs(os.path.dirname(output_glb_path) or ".", exist_ok=True)
    g.save(output_glb_path)

    if allow_skin:
        _assert_skin_intact(output_glb_path)

    stats = {
        "original_size": orig_size,
        "optimized_size": os.path.getsize(output_glb_path),
        "triangles_before": int(tri_before),
        "triangles_after": int(tri_after),
        "vertices_before": int(vert_before),
        "vertices_after": int(vert_after),
        "target_ratio": target_ratio,
        "target_triangles": int(round(target_ratio * tri_before)),
        # Seam fins keep the texture from smearing; on a seam-dense piece (a low-poly kit part where
        # most edges are UV or hard-edge seams) they also stop the reduction above the target, and
        # that is the honest result — the alternative is the smeared decal.
        "seam_lock": bool(seam_lock),
        "seam_limited": bool(seam_lock and tri_after > 1.25 * target_ratio * tri_before),
        "head_boost": head_boost,
        "hand_boost": hand_boost,
        # Share of surviving vertices that lie in the head band — the number the importance warp
        # exists to raise (uniform QEM lands around 0.12-0.21 on a full body, the warp 0.35-0.40).
        "head_vertex_share": (
            float(np.mean([d["head_vertex_share_after"] for d in density
                           if "head_vertex_share_after" in d]))
            if any("head_vertex_share_after" in d for d in density) else None
        ),
        # Quality telemetry. `ancestor_faces` is deliberately NOT included: it is a per-face
        # ndarray and this dict is meant to be JSON-serialized (e.g. into an LOD ladder record).
        "quality": _summarize_quality(density, plans),
        **bake_stats,
    }
    if stats["seam_limited"]:
        logger.warning("retopotool: %s stopped at %d triangles (target %d): its UV / hard-edge "
                       "seams cannot be reduced further without smearing the texture",
                       source_glb_path, tri_after, stats["target_triangles"])
    logger.info("Decimated %s -> %s (%s)", source_glb_path, output_glb_path, stats)
    return stats
