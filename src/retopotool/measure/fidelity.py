"""Measure what an optimize / LOD pass actually cost the character.

    retopo measure fidelity <source.glb> <optimized.glb> [...]

A mesh quality claim needs a measured target, not a heatmap. The complaint this
answers is "the face loses too much resolution and the texture goes smudgy after optimizing", so
the numbers are REGIONAL (head / hands / body) rather than a single mesh-wide average — a mesh-wide
figure is dominated by the torso, which nobody complained about, and it is exactly what let a
face-detail regression ship.

Reported per region:
  p2s p50/p95/max   point-to-surface distance from the SOURCE vertices to the optimized surface,
                    in millimetres. This is the geometry that was thrown away. The head numbers are
                    the ones the complaint is about; the body is the control.
  share             the region's share of surviving vertices — the density allocation.
  normal deg        angular error between the source's shaded normal and what the optimized mesh
                    reconstructs (interpolated vertex normal perturbed by its baked normal map),
                    sampled at the source vertices. This is what the eye actually sees, and it is
                    the only number that can show the normal-map bake doing its job: geometry error
                    can rise while shading error falls.

Compare two candidates by passing several optimized files; they are reported side by side.

Pass `--quality` for the columns point-to-surface distance CANNOT see: triangle shape (slivers,
min-angle, edge-length uniformity), topology (boundary edges, winding consistency), and — the one
that mattered most — whether the two sides of a UV seam still carry the same skin row. A LOD can be
watertight, correctly wound, volume-preserving and sub-millimetre accurate and still tear itself
open along every seam the moment it is posed.

    retopo measure fidelity --quality <one.glb>
    retopo measure fidelity --quality <rig.glb> <lod...>

Pass `--prop` to report one region ("all") instead of the humanoid head / hands / body bands.
Every triangle primitive is measured, placed by its node transform.
"""
from __future__ import annotations

import os
import sys

import numpy as np



def load(path):
    """All triangle primitives of the file in world space, concatenated. The normal map is used
    only when every primitive samples the SAME one (one material, or materials sharing a map) —
    with several, a per-vertex lookup would need a map per primitive, so shading is then compared
    on the interpolated normals alone."""
    from ..bake_normals import decode_image, _texture_image_index_compat
    from ..gltf_io import concat_triangles, load_triangles
    g, blob, prims = load_triangles(path)
    if not prims:
        raise SystemExit(f"no triangle geometry in {path}")
    c = concat_triangles(prims)
    imgs = set()
    for p in prims:
        mat = g.materials[p["material"]] if (p["material"] is not None and g.materials) else None
        nt = mat.normalTexture if mat is not None else None
        imgs.add(_texture_image_index_compat(g, nt.index) if nt is not None else None)
    nmap = None
    if len(imgs) == 1 and None not in imgs:
        ii = imgs.pop()
        if g.images and ii < len(g.images) and g.images[ii].bufferView is not None:
            bv = g.bufferViews[g.images[ii].bufferView]
            off = bv.byteOffset or 0
            nmap = decode_image(bytes(blob[off:off + bv.byteLength]))
    N = c["NORMAL"]
    if N is None:
        from ..bake_normals import _smooth_normals
        N = _smooth_normals(c["P"], c["F"])
    TAN = c["TANGENT"]
    if TAN is None and nmap is not None and c["UV"] is not None:
        # No tangents in the file: a renderer generates them, so the reference must too. Skipping
        # the normal map instead compares a mapped output against an unmapped source.
        from ..bake_normals import _tangents
        TAN = _tangents(c["P"], c["F"], c["UV"], N)
    return c["P"], c["F"], c["UV"], N, TAN, nmap


def regions(P, character=True):
    """Head / hands / body, the same bands the decimator's importance warp uses — or, for a prop,
    one region covering everything."""
    if not character:
        return {"all": np.ones(len(P), dtype=bool)}
    lo, hi = P.min(0), P.max(0)
    h = hi[1] - lo[1]
    head = P[:, 1] > lo[1] + 0.84 * h
    hands = (np.abs(P[:, 0]) > 0.75 * np.abs(P[:, 0]).max()) & ~head
    return {"head": head, "hands": hands, "body": ~head & ~hands}


def shaded_normals(Q, P, F, UV, N, TAN, nmap):
    """The normal the renderer will actually use at the surface point nearest each Q."""
    from ..bake_normals import shade_at
    return shade_at(Q, P, F, UV, N, TAN, nmap)


def quality_block(path, character=True):
    """Triangle shape + topology + skin-seam consistency for one file.

    These are the numbers point-to-surface distance cannot see. A sliver sits right ON the surface
    (tiny p2s) and still shades like a crack; and a skin row that differs between two vertices at the
    SAME position is invisible to every static check yet tears the mesh open once it is posed.
    """
    from ..mesh_quality import (topology_report, triangle_quality, triangle_quality_by_region,
                                skin_discontinuity, dense_weights, weld_by_position)
    from ..gltf_io import concat_triangles, load_triangles
    g, _, prims = load_triangles(path)
    c = concat_triangles(prims)
    P, F = c["P"], c["F"]
    Pw, Fw, _ = weld_by_position(P, F)
    t, q = topology_report(Pw, Fw, welded=True), triangle_quality(Pw, Fw)
    print(f"  topology  bnd={t['boundary_edges']} nonmanifold={t['nonmanifold_edges']} "
          f"dup={t['duplicate_faces']} degen={t['degenerate_faces']} "
          f"winding_inconsistent={t['winding_inconsistent']} vol={t['signed_volume']:.5f}")
    print(f"  shape     slivers<10deg={q['sliver_frac']*100:.2f}%  min-angle p5={q['min_angle_p5']:.2f}deg  "
          f"edge p50={q['edge_p50_mm']:.2f} p99={q['edge_p99_mm']:.2f} max={q['edge_max_mm']:.1f}mm  "
          f"max/p50={q['edge_max_over_p50']:.1f}")
    for r, v in (triangle_quality_by_region(Pw, Fw).items() if character else ()):
        print(f"    {r:6s} tris={v['triangles']:7d} slivers={v['sliver_frac']*100:5.2f}% "
              f"p5={v['min_angle_p5']:5.2f}deg maxedge={v['edge_max_mm']:6.1f}mm")
    if not g.skins or c["JOINTS_0"] is None or c["WEIGHTS_0"] is None:
        return
    J = c["JOINTS_0"]
    W = c["WEIGHTS_0"].astype(np.float64)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    nb = len(g.skins[0].joints)
    sd = skin_discontinuity(P, F, J, W, nb)
    # THE seam check: vertices at one position must carry one skin row.
    M = dense_weights(J, W, nb)
    key = np.round(P * 1e6).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    st = np.searchsorted(inv[order], np.arange(len(cnt)))
    en = np.r_[st[1:], len(order)]
    multi = np.flatnonzero(cnt > 1)
    bad = sum(1 for gi in multi
              if np.abs(M[order[st[gi]:en[gi]]] - M[order[st[gi]]]).sum(1).max() > 1e-6)
    print(f"  skin      seam groups={len(multi)} DIFFERING={bad} "
          f"({100*bad/max(len(multi),1):.1f}%  must be 0)")
    print(f"            edge L1 p95={sd['weight_l1_p95']:.3f} bone-swap={sd['bone_swap_frac']*100:.3f}% "
          f"dominant-change={sd['dominant_change_frac']*100:.2f}% mean-edge={sd['mean_edge_mm']:.2f}mm")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quality = "--quality" in argv
    if quality:
        argv.remove("--quality")
    prop = "--prop" in argv
    if prop:
        argv.remove("--prop")
    if len(argv) < 1:
        raise SystemExit(__doc__)
    if quality and len(argv) == 1:
        print(f"{os.path.basename(argv[0])}")
        quality_block(argv[0], character=not prop)
        return 0
    if len(argv) < 2:
        raise SystemExit(__doc__)
    import igl
    src, cands = argv[0], argv[1:]
    Ps, Fs, UVs, Ns, TANs, NMs = load(src)
    reg = regions(Ps, character=not prop)
    rng = np.random.default_rng(0)
    sample = {k: rng.choice(np.where(m)[0], min(20000, int(m.sum())), replace=False)
              for k, m in reg.items() if m.sum() >= 64}
    ref = {k: shaded_normals(Ps[i], Ps, Fs, UVs, Ns, TANs, NMs) for k, i in sample.items()}
    print(f"source: {os.path.basename(src)}  {len(Fs):,} tris  {len(Ps):,} verts"
          f"  normal map: {'yes' if NMs is not None else 'no'}")
    for cand in cands:
        P, F, UV, N, TAN, NM = load(cand)
        cr = regions(P, character=not prop)
        print(f"\n{os.path.basename(cand)}  {len(F):,} tris  {len(P):,} verts"
              f"  normal map: {'yes' if NM is not None else 'no'}")
        print(f"  {'region':6} {'share':>7} {'p2s p50':>9} {'p2s p95':>9} {'p2s max':>9}"
              f" {'normal p50':>11} {'normal p95':>11}")
        for name, idx in sample.items():
            d2, _, _ = igl.point_mesh_squared_distance(Ps[idx], P, F)
            d = np.sqrt(d2) * 1e3
            n = shaded_normals(Ps[idx], P, F, UV, N, TAN, NM)
            ang = np.degrees(np.arccos(np.clip((n * ref[name]).sum(1), -1, 1)))
            share = cr[name].mean() if name in cr else float("nan")
            print(f"  {name:6} {share*100:6.1f}% {np.median(d):9.3f} {np.percentile(d,95):9.3f}"
                  f" {d.max():9.2f} {np.median(ang):10.2f}° {np.percentile(ang,95):10.2f}°")
        if quality:
            quality_block(cand, character=not prop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
