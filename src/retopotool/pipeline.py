"""High-level entry points: one-call Optimize and a runtime LOD ladder.

`optimize_glb` picks a keep-ratio from the mesh's triangle count and runs the seam-preserving
decimation + normal bake on an UNRIGGED mesh (a prop, an environment piece, a character before
rigging). `build_lod_ladder` writes one LOD per tier of any mesh (skin preserved when there is
one), every rung baked against the full-resolution source.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .decimate import (PROFILES, NothingToDecimate, TARGET_TRIANGLES, decimate_rigged_glb, decimate_source_glb,
                       mesh_stats)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LodTier:
    """One rung of an LOD ladder.

    The budget is either absolute (`triangles`) or relative to the source (`ratio`, the fraction of
    triangles to keep): exactly one of the two. `texture_size` caps the textures shipped with the
    rung and sets the bake resolution; None keeps the source's textures as they are.

        LodTier("far", 5_000, 256)              # 5k triangles, 256px textures
        LodTier("lod2", ratio=0.25)             # a quarter of the source, textures untouched
    """
    name: str
    triangles: int = 0
    texture_size: Optional[int] = None
    ratio: Optional[float] = None

    def budget(self, source_triangles: int) -> int:
        """Triangle target for a source of `source_triangles`."""
        if self.ratio is not None:
            return max(int(round(source_triangles * float(self.ratio))), 1)
        return int(self.triangles)


# Measured on a 38.1 MB rigged character (point-to-surface error vs the full-resolution rig, and
# the size the writer actually produced):
#
#   high    100k / 2048  head 0.62 mm  body 1.28 mm  -> 15.8 MB
#   medium   60k / 1024  head 1.19 mm  body 2.29 mm  ->  5.3 MB
#   low      30k / 1024  head 2.14 mm  body 4.19 mm  ->  3.8 MB
#   minimum  15k /  512  head 4.09 mm  body 7.30 mm  ->  1.7 MB
#
# Every rung re-bakes its normal map from the FULL-RESOLUTION source, which is what keeps the
# shading close as the geometry falls away — at 30k the bake takes body shading error from 10.3 deg
# to 3.4 deg. Without it an LOD is just a coarser mesh.
DEFAULT_LOD_TIERS: List[LodTier] = [
    LodTier("high", 100_000, 2048),
    LodTier("medium", 60_000, 1024),
    LodTier("low", 30_000, 1024),
    LodTier("minimum", 15_000, 512),
]
# The ladder for props and environment pieces, which come in every size from a 300-triangle crate
# to a 2M-triangle scanned cliff, so the budgets are RELATIVE to the source. Textures are left at
# the source size: on a prop they are usually shared with other assets (a trim sheet, a tiling
# material), and a per-rung downscale would only duplicate them.
DEFAULT_PROP_LOD_TIERS: List[LodTier] = [
    LodTier("lod1", ratio=0.5),
    LodTier("lod2", ratio=0.25),
    LodTier("lod3", ratio=0.1),
]
# Below this the source is already at or under a rung and that rung is skipped rather than written
# as a near-copy.
LOD_SKIP_MARGIN = 1.1
# A rung under this many triangles is not written (a prop's 10% rung of a 200-triangle source).
LOD_MIN_TRIS = 12
LOD_MAX_TRIS = 1_000_000
LOD_MIN_RATIO, LOD_MAX_RATIO = 0.01, 0.95
LOD_TEXTURE_SIZES = (256, 512, 1024, 2048, 4096)
# A tier name is also a PATH SEGMENT (`{stem}_lod_{name}.glb`), so it is restricted to a safe
# alphabet rather than sanitized at the point of use.
LOD_NAME_PATTERN = re.compile(r"^[a-z0-9_-]{1,32}$")

# Thresholds for the per-rung warning. These WARN, they never fail a build: a legitimately unusual
# asset (a very flat prop, a mesh with genuine open borders) can exceed them while still being
# exactly what was asked for. The two that indicate real corruption rather than cosmetics — a torn
# hole and a skin that no longer follows position — are asserted inside the decimator itself.
LOD_WARN = {
    "sliver_frac": 0.08,           # measured 0.055 after the quality pass, 0.091 before
    "edge_max_over_p50": 8.0,      # density uniformity
    "winding_inconsistent": 64,    # a handful is the solidity pass doing its job; hundreds is a bug
}


def optimize_glb(source_glb_path: str, output_glb_path: str, *,
                 target_triangles: int = TARGET_TRIANGLES,
                 ratio: Optional[float] = None,
                 bake: bool = True,
                 profile: str = "prop",
                 head_boost: Optional[float] = None,
                 hand_boost: Optional[float] = None) -> Dict:
    """Optimize an UNRIGGED GLB in one call.

    With `ratio=None` the keep-ratio is derived from `target_triangles`: a mesh above it is brought
    down to it (ratio clamped to 0.05-0.95); a mesh already at or under it still gets a light 5%
    pass. Pass `ratio` (0.05-0.95) to choose it yourself. `profile` is "prop" (uniform budget,
    the default) or "character" (head and hands keep more of it). Never overwrites the input.
    Raises ValueError for skinned / animated / morphed inputs.
    """
    if ratio is None:
        stats = mesh_stats(source_glb_path)
        if not stats.get("optimizable"):
            raise ValueError("mesh is not optimizable (no triangle geometry, or it is "
                             "rigged/animated — use build_lod_ladder for rigged meshes)")
        tris = int(stats["triangles"])
        ratio = (min(max(round(target_triangles / float(tris), 3), 0.05), 0.95)
                 if tris > target_triangles else 0.95)
    elif not 0.05 <= float(ratio) <= 0.95:
        raise ValueError(f"ratio must be within 0.05-0.95, got {ratio}")
    if os.path.abspath(source_glb_path) == os.path.abspath(output_glb_path):
        raise ValueError("output path must differ from the source (the optimizer never overwrites "
                         "its input)")
    return decimate_source_glb(source_glb_path, output_glb_path, float(ratio), bake=bake,
                               profile=profile, head_boost=head_boost, hand_boost=hand_boost)


def merge_lod_levels(existing: Optional[List[Dict[str, Any]]],
                     built: List[Dict[str, Any]],
                     removed: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Fold freshly built rungs into an existing ladder record.

    MERGE, not replace: entries are keyed by `name` — a rebuild replaces its own entry and leaves the
    rest alone. `level` and `screen_pct` are RECOMPUTED over the merged set ranked by triangle count
    descending, so level 0 is always the heaviest rung. Entries with no `name` are dropped.
    """
    drop = set(removed or ())
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in list(existing or []) + list(built):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in drop:
            continue
        by_name[name] = dict(entry)
    # sorted() is stable, so equal triangle counts keep the order they were seen in.
    merged = sorted(by_name.values(), key=lambda e: -int(e.get("triangles") or 0))
    for level, entry in enumerate(merged):
        entry["level"] = level
        # Fraction of screen height below which this rung is appropriate — a sensible default.
        entry["screen_pct"] = round(1.0 / (2 ** level), 3)
    return merged


def warn_lod_quality(name: str, q: Dict) -> List[str]:
    """Log (and return) anything outside the expected band for a healthy rung."""
    warnings: List[str] = []
    if not q:
        return warnings
    for key, limit in LOD_WARN.items():
        val = q.get(key)
        if val is not None and val > limit:
            warnings.append(f"{key}={val} exceeds {limit}")
    if q.get("boundary_edges"):
        warnings.append(f"introduced {q['boundary_edges']} boundary edges — the mesh was torn")
    if q.get("skin_keyed_on_position") is False:
        warnings.append("skin resolved per UV wedge; seams will tear")
    for w in warnings:
        logger.warning("lod tier %r: %s", name, w)
    return warnings


def _validate_tier(tier: LodTier) -> None:
    if not LOD_NAME_PATTERN.match(tier.name or ""):
        raise ValueError(f"invalid tier name {tier.name!r}: use 1-32 chars of [a-z0-9_-]")
    if tier.ratio is not None:
        if tier.triangles:
            raise ValueError(f"tier {tier.name!r}: give triangles OR ratio, not both")
        if not LOD_MIN_RATIO <= float(tier.ratio) <= LOD_MAX_RATIO:
            raise ValueError(f"tier {tier.name!r}: ratio must be within "
                             f"{LOD_MIN_RATIO}-{LOD_MAX_RATIO}, got {tier.ratio}")
    elif not LOD_MIN_TRIS <= int(tier.triangles) <= LOD_MAX_TRIS:
        raise ValueError(f"tier {tier.name!r}: triangles must be within "
                         f"{LOD_MIN_TRIS}-{LOD_MAX_TRIS}, got {tier.triangles}")
    if tier.texture_size is not None and int(tier.texture_size) not in LOD_TEXTURE_SIZES:
        raise ValueError(f"tier {tier.name!r}: texture_size must be one of {LOD_TEXTURE_SIZES}, "
                         f"got {tier.texture_size}")


def build_lod_ladder(source_glb_path: str, out_dir: str,
                     tiers: Optional[Iterable[LodTier]] = None, *,
                     stem: Optional[str] = None,
                     bake: bool = True,
                     profile: str = "auto",
                     head_boost: Optional[float] = None,
                     hand_boost: Optional[float] = None) -> Dict[str, Any]:
    """Write `{out_dir}/{stem}_lod_{tier}.glb` for every tier and return the ladder.

    `tiers=None` picks the default ladder for the input: `DEFAULT_LOD_TIERS` (absolute character
    budgets) for a skinned GLB, `DEFAULT_PROP_LOD_TIERS` (50% / 25% / 10%) otherwise. `profile`
    is passed to the decimator ("auto" = humanoid warp only for a skinned GLB).

    A tier is SKIPPED when the source has no more than `LOD_SKIP_MARGIN` x its triangle budget, or
    when its budget falls under `LOD_MIN_TRIS`.
    Each rung is written to a hidden temp file and atomically renamed into place, so a half-written
    GLB is never visible at the real path. Every rung bakes its normal map from the
    full-resolution source (not the rung above). Raises ValueError on an invalid tier or an input
    the decimator refuses; the temp file is removed either way.

    Returns {"source_triangles", "levels": [entry...], "skipped": [name...]} where each entry has
    name, file, target_triangles, triangles, vertices, texture_size, size_bytes, quality, warnings,
    level and screen_pct (level 0 = heaviest).
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
    if tiers is None:
        tiers = (DEFAULT_LOD_TIERS if mesh_stats(source_glb_path)["has_skin"]
                 else DEFAULT_PROP_LOD_TIERS)
    tiers = list(tiers)
    for tier in tiers:
        _validate_tier(tier)
    if len({t.name for t in tiers}) != len(tiers):
        raise ValueError("tier names must be unique")
    src = Path(source_glb_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or src.stem
    src_tris = int(mesh_stats(str(src))["triangles"])

    levels: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for tier in tiers:
        budget = tier.budget(src_tris)
        if src_tris <= budget * LOD_SKIP_MARGIN or budget < LOD_MIN_TRIS:
            logger.info("lod: skipping %r (source is %d tris, budget %d)", tier.name, src_tris,
                        budget)
            skipped.append(tier.name)
            continue
        out_path = out / f"{stem}_lod_{tier.name}.glb"
        # The temp name MUST still end in `.glb`: pygltflib picks GLB vs JSON off the extension, so
        # a `.glb.tmp` target is silently written as a JSON glTF + `.bin` sidecar. Hidden + stamped
        # so concurrent builds of the same tier cannot collide.
        tmp_path = out_path.with_name(f".{out_path.stem}_{uuid.uuid4().hex[:8]}.tmp.glb")
        try:
            stats = decimate_rigged_glb(str(src), str(tmp_path), budget,
                                        texture_size=tier.texture_size, bake=bake,
                                        bake_reference=str(src), profile=profile,
                                        head_boost=head_boost, hand_boost=hand_boost)
            os.replace(tmp_path, out_path)
        except NothingToDecimate:
            logger.info("lod: skipping %r (nothing large enough to reduce)", tier.name)
            skipped.append(tier.name)
            continue
        finally:
            tmp_path.unlink(missing_ok=True)
        quality = stats.get("quality") or {}
        levels.append({
            "name": tier.name,
            "file": str(out_path),
            # What was ASKED for, kept beside what was achieved: decimation lands near the target,
            # not on it.
            "target_triangles": int(budget),
            "triangles": int(stats["triangles_after"]),
            "vertices": int(stats["vertices_after"]),
            "texture_size": int(tier.texture_size) if tier.texture_size else None,
            "size_bytes": out_path.stat().st_size,
            "normal_map_baked": bool(stats.get("normal_map_baked")),
            "bake_skipped": stats.get("bake_skipped") or [],
            "quality": quality,
            "warnings": warn_lod_quality(tier.name, quality),
        })
    return {"source_triangles": src_tris,
            "levels": merge_lod_levels(None, levels),
            "skipped": skipped}
