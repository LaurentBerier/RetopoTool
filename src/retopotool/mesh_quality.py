"""Mesh quality metrics for the character decimation / LOD pipeline.

Pure, dependency-light (numpy only) measurements shared by three callers that MUST agree on the
numbers, or a "quality gate" means something different from what the instrument printed:

  * `retopotool.measure.fidelity` — the offline instrument used to accept a change
  * `retopotool.decimate`         — the per-rung `quality` telemetry every decimation returns
  * `tests/`                      — the regression assertions

WHY THESE METRICS AND NOT THE OBVIOUS ONES. Measured on a real armored character (Char_Carnage,
1.83 m, 400,934 tris) decimated to the shipped LOD ladder, the mesh is WATERTIGHT at every rung
(0 boundary edges), consistently wound, volume-preserving to 0.6%, and its point-to-surface error
is sub-millimetre — and it still looked broken. Point-to-surface distance is a near-useless
acceptance test on its own because a SLIVER triangle is geometrically close to the surface while
shading terribly: what degrades is triangle SHAPE, and it degrades non-uniformly.

  region (LOD low, 30k)   slivers <10deg   min-angle p5   max edge
  head                          5.31%          9.81deg      52.9 mm
  torso                        10.07%          7.76deg      90.1 mm
  legs                         10.11%          7.68deg     151.9 mm

So `triangle_quality` is the headline instrument, `topology_report` is the corruption check (a
regression here is a real hole, not a cosmetic one), and `skin_discontinuity` catches the defect
that only appears once the mesh is POSED — LOD weights are piecewise-constant, and at a 16 mm edge
a single collapse can swap bones outright.

Every function takes an already-welded (positions, faces) pair unless stated. Weld with `weld_by_position`
first: an un-welded mesh reports every UV seam as a boundary edge, which is the classic false alarm.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# A triangle whose smallest angle is under this is a "sliver": its vertex normals interpolate over a
# near-degenerate footprint, which on a specular material reads as a bright/dark streak or a crack.
SLIVER_ANGLE_DEG = 10.0
# Dihedral above which an edge is a hard-surface CREASE (an armor plate boundary) rather than a
# smooth curvature edge. Used by the quality pass to decide what it is allowed to move or flip.
CREASE_ANGLE_DEG = 25.0


def weld_by_position(P: np.ndarray, F: np.ndarray, tol: Optional[float] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge bit-coincident vertices so topology can be measured.

    Returns (welded positions, welded faces, raw->welded index map). `tol` defaults to the same
    span-relative quantization `decimate._weld` uses, so a measurement here matches what the
    decimator itself considered one vertex.
    """
    P = np.asarray(P, dtype=np.float64)
    if tol is None:
        span = float(np.linalg.norm(P.max(0) - P.min(0))) if len(P) else 1.0
        tol = max(span, 1e-6) * 1e-6
    Pq = np.round(P / tol).astype(np.int64)
    uniq, first, inv = np.unique(Pq, axis=0, return_index=True, return_inverse=True)
    return P[first], inv[np.asarray(F, dtype=np.int64)], inv.astype(np.int64)


def _undirected_edges(F: np.ndarray) -> np.ndarray:
    return np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0), axis=1)


def face_normals(P: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unit face normals and areas. A degenerate face gets a ZERO normal (not a NaN) so callers can
    filter it with a norm test rather than propagating NaNs into an average."""
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    area = 0.5 * ln
    ok = ln > 1e-20
    out = np.zeros_like(n)
    out[ok] = n[ok] / ln[ok][:, None]
    return out, area


def topology_report(P: np.ndarray, F: np.ndarray, welded: bool = False) -> Dict:
    """Corruption check. A regression in any of these is a REAL defect, not a cosmetic one.

    `boundary_edges` on a closed character means the decimator tore a hole. `winding_inconsistent`
    means neighbouring faces disagree on which side is out — with a single-sided material that is an
    invisible face, i.e. a see-through hole.
    """
    P = np.asarray(P, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if not welded:
        P, F, _ = weld_by_position(P, F)
    degen = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
    Fk = F[~degen]
    E = _undirected_edges(Fk)
    _, counts = np.unique(E, axis=0, return_counts=True)
    fs = np.sort(Fk, axis=1)
    _, fcounts = np.unique(fs, axis=0, return_counts=True)

    # Winding consistency: a shared edge must be traversed in OPPOSITE directions by its two faces.
    DE = np.concatenate([Fk[:, [0, 1]], Fk[:, [1, 2]], Fk[:, [2, 0]]], axis=0)
    sm = np.sort(DE, axis=1)
    flipped = DE[:, 0] != sm[:, 0]
    code = sm[:, 0].astype(np.int64) * (len(P) + 1) + sm[:, 1]
    order = np.argsort(code, kind="stable")
    cs, fsort = code[order], flipped[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    runs = np.diff(np.r_[starts, len(cs)])
    pair = starts[runs == 2]
    inconsistent = int((fsort[pair] == fsort[pair + 1]).sum()) if len(pair) else 0

    _, area = face_normals(P, Fk)
    a, b, c = P[Fk[:, 0]], P[Fk[:, 1]], P[Fk[:, 2]]
    volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    return {
        "vertices": int(len(P)),
        "triangles": int(len(F)),
        "degenerate_faces": int(degen.sum()),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "duplicate_faces": int((fcounts > 1).sum()),
        "winding_inconsistent": inconsistent,
        "signed_volume": volume,
        "zero_area_faces": int((area < 1e-12).sum()),
    }


def triangle_quality(P: np.ndarray, F: np.ndarray) -> Dict:
    """Triangle SHAPE and size distribution — the headline retopology instrument.

    `sliver_frac` and `min_angle_p5` say how badly shaped the worst triangles are; the edge-length
    percentiles say how UNIFORM the density is. `edge_max / edge_p50` is the number that exposed the
    body starvation: 11x at LOD low, i.e. a single triangle spanning most of a limb.
    """
    P = np.asarray(P, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    e0 = np.linalg.norm(b - a, axis=1)
    e1 = np.linalg.norm(c - b, axis=1)
    e2 = np.linalg.norm(a - c, axis=1)

    def angle(opp, x, y):
        return np.degrees(np.arccos(np.clip((x * x + y * y - opp * opp) / (2 * np.maximum(x * y, 1e-15)),
                                            -1.0, 1.0)))

    amin = np.minimum(np.minimum(angle(e0, e1, e2), angle(e1, e2, e0)), angle(e2, e0, e1))
    L = np.stack([e0, e1, e2], axis=1)
    lmax, lmin = L.max(1), L.min(1)
    all_e = np.concatenate([e0, e1, e2])
    aspect = lmax / np.maximum(lmin, 1e-15)
    return {
        "sliver_frac": float((amin < SLIVER_ANGLE_DEG).mean()),
        "min_angle_p1": float(np.percentile(amin, 1)),
        "min_angle_p5": float(np.percentile(amin, 5)),
        "min_angle_p50": float(np.median(amin)),
        "edge_p50_mm": float(np.median(all_e) * 1e3),
        "edge_p99_mm": float(np.percentile(all_e, 99) * 1e3),
        "edge_max_mm": float(all_e.max() * 1e3),
        "edge_max_over_p50": float(all_e.max() / max(np.median(all_e), 1e-15)),
        "aspect_p99": float(np.percentile(aspect, 99)),
        "aspect_max": float(aspect.max()),
    }


def region_masks(P: np.ndarray, head_frac: float = 0.86, torso_frac: float = 0.55) -> Dict[str, np.ndarray]:
    """Head / torso / legs by height fraction. Deliberately the SAME bands the reports above use so
    a regional claim is comparable across runs. (`decimate` boosts above 0.84; 0.86 here keeps
    the head band clear of the warp's ramp so the measurement is not reading the ramp itself.)"""
    P = np.asarray(P, dtype=np.float64)
    lo, hi = P.min(0), P.max(0)
    h = float(hi[1] - lo[1]) or 1.0
    hf = (P[:, 1] - lo[1]) / h
    return {"head": hf > head_frac,
            "torso": (hf > torso_frac) & (hf <= head_frac),
            "legs": hf <= torso_frac}


def triangle_quality_by_region(P: np.ndarray, F: np.ndarray) -> Dict[str, Dict]:
    """`triangle_quality` per body region, keyed on each triangle's CENTROID height.

    A mesh-wide figure is dominated by the legs and hides the thing the user actually reports; the
    face/body split only shows up regionally.
    """
    P = np.asarray(P, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    cent = (P[F[:, 0]] + P[F[:, 1]] + P[F[:, 2]]) / 3.0
    masks = region_masks(cent)
    out = {}
    for name, m in masks.items():
        if int(m.sum()) < 16:
            continue
        out[name] = triangle_quality(P, F[m])
        out[name]["triangles"] = int(m.sum())
    return out


def dense_weights(J: np.ndarray, W: np.ndarray, n_bones: int) -> np.ndarray:
    """(n_verts, n_bones) weight matrix from the 4-influence glTF pair, rows normalized to sum 1.

    `J` MUST be the raw integer joint indices. glTF stores JOINTS_0 as UNSIGNED_BYTE with
    normalized=false — a generic accessor reader that divides by dtype max turns joint 66 into 0.259
    and every metric built on it is silently meaningless.
    """
    J = np.asarray(J, dtype=np.int64)
    W = np.asarray(W, dtype=np.float64)
    M = np.zeros((len(J), n_bones), dtype=np.float64)
    rows = np.arange(len(J))
    for k in range(J.shape[1]):
        np.add.at(M, (rows, J[:, k]), W[:, k])
    return M / np.maximum(M.sum(1, keepdims=True), 1e-12)


def skin_discontinuity(P: np.ndarray, F: np.ndarray, J: np.ndarray, W: np.ndarray,
                       n_bones: int) -> Dict:
    """How abruptly skin weights change across a mesh edge — the defect that only shows when POSED.

    Decimation copies each output vertex's weights verbatim from one source vertex, so as edges grow
    the weight field stops being a smooth partition and starts stepping. `bone_swap_frac` counts
    edges whose two ends share almost no influence at all (L1 > 1.0): under animation those two
    vertices travel with different limbs and the triangle between them stretches into a visible flap.
    Measured: 0.023% on the 400k rig, 0.797% at LOD low — a 35x rate increase.
    """
    F = np.asarray(F, dtype=np.int64)
    M = dense_weights(J, W, n_bones)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    d = np.abs(M[E[:, 0]] - M[E[:, 1]]).sum(1)
    dom = M.argmax(1)
    P = np.asarray(P, dtype=np.float64)
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    return {
        "weight_l1_p50": float(np.median(d)),
        "weight_l1_p95": float(np.percentile(d, 95)),
        "weight_l1_p99": float(np.percentile(d, 99)),
        "weight_l1_max": float(d.max()),
        "bone_swap_frac": float((d > 1.0).mean()),
        "bone_swap_edges": int((d > 1.0).sum()),
        "hard_tear_edges": int((d > 1.5).sum()),
        "dominant_change_frac": float((dom[E[:, 0]] != dom[E[:, 1]]).mean()),
        "mean_edge_mm": float(L.mean() * 1e3),
    }


def crease_edges(P: np.ndarray, F: np.ndarray, angle_deg: float = CREASE_ANGLE_DEG
                 ) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], np.ndarray]:
    """Manifold interior edges and which of them are hard-surface creases.

    Returns (edge -> (face_a, face_b) for edges with exactly two incident faces, boolean crease mask
    aligned with that dict's insertion order). The crease test is the dihedral between the two face
    normals; it is what the quality pass uses to decide what it may move or flip, and it is the
    single parameter trading hard-surface fidelity against sliver removal.
    """
    F = np.asarray(F, dtype=np.int64)
    N, _ = face_normals(np.asarray(P, dtype=np.float64), F)
    E = _undirected_edges(F)
    fid = np.tile(np.arange(len(F)), 3)
    code = E[:, 0].astype(np.int64) * (int(F.max()) + 2) + E[:, 1]
    order = np.argsort(code, kind="stable")
    cs, es, fsorted = code[order], E[order], fid[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    runs = np.diff(np.r_[starts, len(cs)])
    two = starts[runs == 2]
    pairs = {}
    cos_thr = np.cos(np.radians(angle_deg))
    mask = np.zeros(len(two), dtype=bool)
    for i, s in enumerate(two):
        f0, f1 = int(fsorted[s]), int(fsorted[s + 1])
        pairs[(int(es[s, 0]), int(es[s, 1]))] = (f0, f1)
        mask[i] = float(N[f0] @ N[f1]) < cos_thr
    return pairs, mask
