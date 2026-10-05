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
"""
from __future__ import annotations

import os
import sys

import numpy as np
from pygltflib import GLTF2

from ..gltf_io import _acc


def load(path):
    g = GLTF2().load(path)
    blob = g.binary_blob()
    P = F = UV = N = TAN = None
    for mesh in (g.meshes or []):
        for prim in mesh.primitives:
            if prim.attributes.POSITION is None:
                continue
            P = _acc(g, blob, prim.attributes.POSITION).astype(np.float64)
            F = (_acc(g, blob, prim.indices).astype(np.int64).reshape(-1, 3)
                 if prim.indices is not None else np.arange(len(P)).reshape(-1, 3))
            for name, dst in (("TEXCOORD_0", "UV"), ("NORMAL", "N"), ("TANGENT", "TAN")):
                idx = getattr(prim.attributes, name, None)
                if idx is not None:
                    v = _acc(g, blob, idx).astype(np.float64)
                    if dst == "UV":
                        UV = v
                    elif dst == "N":
                        N = v
                    else:
                        TAN = v
            mat = g.materials[prim.material] if (prim.material is not None and g.materials) else None
            nmap = None
            if mat is not None and mat.normalTexture is not None:
                from ..bake_normals import decode_image, _texture_image_index_compat
                ii = _texture_image_index_compat(g, mat.normalTexture.index)
                if ii is not None and g.images and ii < len(g.images):
                    im = g.images[ii]
                    if im.bufferView is not None:
                        bv = g.bufferViews[im.bufferView]
                        off = bv.byteOffset or 0
                        nmap = decode_image(bytes(blob[off:off + bv.byteLength]))
            return P, F, UV, N, TAN, nmap
    raise SystemExit(f"no triangle geometry in {path}")


def regions(P):
    """Head / hands / body, the same bands the decimator's importance warp uses."""
    lo, hi = P.min(0), P.max(0)
    h = hi[1] - lo[1]
    head = P[:, 1] > lo[1] + 0.84 * h
    hands = (np.abs(P[:, 0]) > 0.75 * np.abs(P[:, 0]).max()) & ~head
    return {"head": head, "hands": hands, "body": ~head & ~hands}


def shaded_normals(Q, P, F, UV, N, TAN, nmap):
    """The normal the renderer will actually use at the surface point nearest each Q."""
    import igl
    _, fid, pts = igl.point_mesh_squared_distance(Q, P, F)
    fid = np.asarray(fid, dtype=np.int64)
    tri = P[F[fid]]
    v0, v1, v2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], np.asarray(pts) - tri[:, 0]
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-20, 1e-20, d00 * d11 - d01 * d01)
    b = (d11 * d20 - d01 * d21) / den
    c = (d00 * d21 - d01 * d20) / den
    w = np.stack([1 - b - c, b, c], axis=1)
    cor = F[fid]
    n = (N[cor] * w[:, :, None]).sum(1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
    if nmap is not None and TAN is not None and UV is not None:
        from ..bake_normals import _sample_bilinear
        t = (TAN[cor, :3] * w[:, :, None]).sum(1)
        t -= n * (n * t).sum(1, keepdims=True)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-20
        bt = np.cross(n, t) * np.sign((TAN[cor, 3] * w).sum(1))[:, None]
        ts = _sample_bilinear(nmap, (UV[cor] * w[:, :, None]).sum(1)) * 2 - 1
        n = t * ts[:, 0:1] + bt * ts[:, 1:2] + n * ts[:, 2:3]
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
    return n


def quality_block(path):
    """Triangle shape + topology + skin-seam consistency for one file.

    These are the numbers point-to-surface distance cannot see. A sliver sits right ON the surface
    (tiny p2s) and still shades like a crack; and a skin row that differs between two vertices at the
    SAME position is invisible to every static check yet tears the mesh open once it is posed.
    """
    from ..mesh_quality import (topology_report, triangle_quality, triangle_quality_by_region,
                                skin_discontinuity, dense_weights, weld_by_position)
    g = GLTF2().load(path)
    blob = g.binary_blob()
    prim = g.meshes[0].primitives[0]
    P = _acc(g, blob, prim.attributes.POSITION).astype(np.float64)
    F = _acc(g, blob, prim.indices).astype(np.int64).reshape(-1, 3)
    Pw, Fw, _ = weld_by_position(P, F)
    t, q = topology_report(Pw, Fw, welded=True), triangle_quality(Pw, Fw)
    print(f"  topology  bnd={t['boundary_edges']} nonmanifold={t['nonmanifold_edges']} "
          f"dup={t['duplicate_faces']} degen={t['degenerate_faces']} "
          f"winding_inconsistent={t['winding_inconsistent']} vol={t['signed_volume']:.5f}")
    print(f"  shape     slivers<10deg={q['sliver_frac']*100:.2f}%  min-angle p5={q['min_angle_p5']:.2f}deg  "
          f"edge p50={q['edge_p50_mm']:.2f} p99={q['edge_p99_mm']:.2f} max={q['edge_max_mm']:.1f}mm  "
          f"max/p50={q['edge_max_over_p50']:.1f}")
    for r, v in triangle_quality_by_region(Pw, Fw).items():
        print(f"    {r:6s} tris={v['triangles']:7d} slivers={v['sliver_frac']*100:5.2f}% "
              f"p5={v['min_angle_p5']:5.2f}deg maxedge={v['edge_max_mm']:6.1f}mm")
    if not g.skins or getattr(prim.attributes, "JOINTS_0", None) is None:
        return
    J = _acc(g, blob, prim.attributes.JOINTS_0).astype(np.int64)
    W = _acc(g, blob, prim.attributes.WEIGHTS_0).astype(np.float64)
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
    if len(argv) < 1:
        raise SystemExit(__doc__)
    if quality and len(argv) == 1:
        print(f"{os.path.basename(argv[0])}")
        quality_block(argv[0])
        return 0
    if len(argv) < 2:
        raise SystemExit(__doc__)
    import igl
    src, cands = argv[0], argv[1:]
    Ps, Fs, UVs, Ns, TANs, NMs = load(src)
    reg = regions(Ps)
    rng = np.random.default_rng(0)
    sample = {k: rng.choice(np.where(m)[0], min(20000, int(m.sum())), replace=False)
              for k, m in reg.items() if m.sum() >= 64}
    ref = {k: shaded_normals(Ps[i], Ps, Fs, UVs, Ns, TANs, NMs) for k, i in sample.items()}
    print(f"source: {os.path.basename(src)}  {len(Fs):,} tris  {len(Ps):,} verts"
          f"  normal map: {'yes' if NMs is not None else 'no'}")
    for cand in cands:
        P, F, UV, N, TAN, NM = load(cand)
        cr = regions(P)
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
            quality_block(cand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
