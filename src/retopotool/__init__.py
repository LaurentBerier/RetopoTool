"""retopotool — seam-preserving GLB decimation, normal-map bake and LOD ladders for props,
environment pieces and characters.

    from retopotool import optimize_glb, build_lod_ladder, mesh_stats

    mesh_stats("crate.glb")                                  # cheap triangle/vertex/skin report
    optimize_glb("rock_scan.glb", "rock.glb", target_triangles=20_000)   # + normal bake
    build_lod_ladder("crate.glb", "lods/")                   # static: 50% / 25% / 10%
    build_lod_ladder("hero_rigged.glb", "lods/")             # skinned: high/medium/low/minimum
"""
from . import mesh_quality
from .bake_normals import bake_normal_map
from .decimate import (DENSE_TRIANGLE_THRESHOLD, PROFILES, TARGET_TRIANGLES, NothingToDecimate,
                       build_skin_proxy, decimate_rigged_glb, decimate_source_glb, mesh_stats)
from .pipeline import (DEFAULT_LOD_TIERS, DEFAULT_PROP_LOD_TIERS, LodTier, build_lod_ladder,
                       merge_lod_levels, optimize_glb, warn_lod_quality)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_LOD_TIERS",
    "DEFAULT_PROP_LOD_TIERS",
    "DENSE_TRIANGLE_THRESHOLD",
    "LodTier",
    "NothingToDecimate",
    "PROFILES",
    "TARGET_TRIANGLES",
    "bake_normal_map",
    "build_lod_ladder",
    "build_skin_proxy",
    "decimate_rigged_glb",
    "decimate_source_glb",
    "merge_lod_levels",
    "mesh_quality",
    "mesh_stats",
    "optimize_glb",
    "warn_lod_quality",
]
