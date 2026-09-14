import math
import time

import bpy  # type: ignore
import numpy as np  # type: ignore
from shapely.geometry import Point  # type: ignore

from ... import constants as const
from ... import progress as _progress
from .. import geometry2d as g2d
from ..mesh_ops import recalculateNormals
from ..scene import remove_objects
from .buildings import _append_building, _make_terrain_sampler, safe_float_height
from .fetch_solo import fetch_osm_data

# Set after each landmarks-enabled generation; read by the puzzle flow to clip
# per-landmark footprints against each jigsaw piece (landmarks are otherwise
# a single standalone object never cut along the puzzle's own seams).
_puzzle_landmarks_data: tuple | None = None

# ---------------------------------------------------------------------------
# OSM tag helpers (mirrored in fetch_solo / fetch_group so the fetch query,
# the combined-fetch classifier and the geometry builder stay in sync).
# ---------------------------------------------------------------------------

_AREA_LANDMARK_TAGS = {
    "tourism": {"attraction", "artwork"},
    "historic": {
        "monument", "memorial", "castle", "ruins", "fort", "manor",
        "tower", "city_gate", "arch",
    },
    "man_made": {"lighthouse", "water_tower", "obelisk"},
    "leisure": {"stadium"},
    "building": {"cathedral", "basilica"},
    "amenity": {"fountain"},
    "natural": {"peak"},
}


def has_landmark_tags(tags):
    """True if an OSM element carries one of the landmark tag combinations."""
    if not tags:
        return False
    for key, values in _AREA_LANDMARK_TAGS.items():
        if tags.get(key) in values:
            return True
    if tags.get("man_made") == "tower":
        # Only towers with a real height tag or a distinct tower:type qualify
        # (utility masts and street furniture do not).
        if tags.get("tower:type"):
            return True
        try:
            return float(str(tags.get("height", "")).split(" ")[0]) >= 20.0
        except (ValueError, TypeError):
            return False
    return False


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _append_node_landmark(x, y, z_offset, base_radius, sample_z, b_verts, b_faces):
    """Synthesize a small circular footprint around a node landmark and build
    it as a pyramid-roofed prism (marker/spire look)."""
    poly = Point(x, y).buffer(max(base_radius, 0.01))
    _append_building(
        poly, z_offset, sample_z, b_verts, b_faces, roof="PYRAMID", roof_frac=0.25
    )


def landmarks_geometry_for_polygon(piece_polygon, landmarks_data):
    """Return (verts, faces) for every landmark footprint clipped to *piece_polygon*.

    `landmarks_data` is the (footprints, sample_z) tuple cached in
    `_puzzle_landmarks_data` by create_landmarks during the puzzle blank's own
    generation. Mirrors buildings_geometry_for_polygon.
    """
    footprints, sample_z = landmarks_data
    b_verts, b_faces = [], []
    for item in footprints:
        poly, z_offset, *roof_info = item
        roof = roof_info[0] if roof_info else "FLAT"
        roof_frac = roof_info[1] if len(roof_info) > 1 else 0.15
        clipped = g2d.validate(poly.intersection(piece_polygon))
        if clipped is None or clipped.is_empty:
            continue
        for part in g2d.iter_polygons(clipped):
            _append_building(
                part, z_offset, sample_z, b_verts, b_faces,
                roof=roof, roof_frac=roof_frac,
            )
    if not b_verts:
        return None, None
    return b_verts, b_faces


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def create_landmarks(map, default_height=10, scaleHor=1.0):
    """Fetch OSM landmark data and build a single 'Landmarks' mesh object.

    Area landmarks (stadiums, castles, cathedrals...) become extruded prisms
    with heights from OSM tags. Node-only landmarks (attractions, monuments,
    towers without footprints) become small pyramid-roofed markers centered on
    their GPS position. Mirrors create_buildings closely.
    """
    _sScaleHor = bpy.context.scene.tp3d.sScaleHor
    _t_setup = time.time()

    wall_obj, _sample_z = _make_terrain_sampler(map)

    _ov = _progress.ProgressOverlay.get()
    if _ov.active:
        _ov.set_fetch_progress("landmarks", 0.0)

    tp3d = bpy.context.scene.tp3d
    minLat = tp3d.minLat
    minLon = tp3d.minLon
    maxLat = tp3d.maxLat
    maxLon = tp3d.maxLon

    b_verts = []
    b_faces = []
    _puzzle_footprints = []
    l_height_mult = tp3d.el_lHeightMultiplier
    node_base_radius = max(0.01, tp3d.el_lBaseMM / 2.0)
    node_min_h_mm = max(0.0, tp3d.el_lNodeHeightMM)
    max_nodes = max(0, int(tp3d.el_lMaxNodes))

    map_fp = g2d.map_footprint_polygon(map)
    print(
        f"[TP3D landmarks] setup (wall extrude + BVH + map outline) took {time.time() - _t_setup:.1f}s"
    )
    if _ov.active:
        _ov.set_fetch_progress("landmarks", 0.15)

    min_area = tp3d.el_bMinPrintMM**2

    lat_step = min(2.0, maxLat - minLat)
    lon_step = min(2.0, maxLon - minLon)
    lats = math.ceil((maxLat - minLat) / lat_step)
    lons = math.ceil((maxLon - minLon) / lon_step)

    # Node landmarks are collected per tile, then globally culled (cap + map
    # footprint containment) before geometry is built, so the same cap applies
    # no matter how tiles split the area.
    pending_nodes = []  # (height_m, x, y)

    if lats * lons < 20:
        for k in range(lats):
            for l in range(lons):
                _cntr = k * lons + l + 1
                _maxcntr = lats * lons
                print(f"Landmarks loop: {_cntr}/{_maxcntr}")
                if _ov.active:
                    _ov.update(
                        message=f"Landmarks: tile {_cntr}/{_maxcntr} — processing…"
                    )
                south = minLat + k * lat_step
                north = south + lat_step
                west = minLon + l * lon_step
                east = west + lon_step

                data = fetch_osm_data((south, west, north, east), "LANDMARKS")
                if not data or "elements" not in data:
                    print("No Landmark data returned")
                    continue

                assert isinstance(data, dict)
                n_landmarks = len(
                    [e for e in data["elements"] if e["type"] != "node"]
                )
                if _ov.active:
                    _ov.update(
                        message=f"Landmarks: tile {_cntr}/{_maxcntr} — calculating {n_landmarks} landmarks…"
                    )

                raw_nodes = {
                    n["id"]: (n["lat"], n["lon"])
                    for n in data["elements"]
                    if n["type"] == "node"
                }

                node_xy = {}
                if raw_nodes:
                    nid_list = list(raw_nodes.keys())
                    arr = np.array(
                        [raw_nodes[nid] for nid in nid_list], dtype=np.float64
                    )  # (N, 2) lat, lon
                    xs = const.R * np.radians(arr[:, 1]) * _sScaleHor
                    ys = (
                        const.R
                        * np.log(np.tan(np.pi / 4.0 + np.radians(arr[:, 0]) / 2.0))
                        * _sScaleHor
                    )
                    for nid, x, y, (nlat, nlon) in zip(
                        nid_list, xs.tolist(), ys.tolist(), arr.tolist()
                    ):
                        node_xy[nid] = (x, y, nlat, nlon)

                ways_by_id = {
                    e["id"]: e for e in data["elements"] if e["type"] == "way"
                }

                _tile_total = max(1, len(data["elements"]))
                for i, element in enumerate(data["elements"]):
                    if _ov.active and i % max(1, _tile_total // 20) == 0:
                        _elem_frac = ((_cntr - 1) + i / _tile_total) / _maxcntr
                        _ov.set_fetch_progress("landmarks", 0.15 + 0.60 * _elem_frac)

                    if element["type"] == "node":
                        tags = element.get("tags", {}) or {}
                        # Skip untagged reference nodes (the '>' recurse adds
                        # every node referenced by the matched ways).
                        if not has_landmark_tags(tags):
                            continue
                        nlat = element.get("lat")
                        nlon = element.get("lon")
                        if nlat is None or nlon is None:
                            continue
                        x = const.R * np.radians(nlon) * _sScaleHor
                        y = (
                            const.R
                            * np.log(np.tan(np.pi / 4.0 + np.radians(nlat) / 2.0))
                            * _sScaleHor
                        )
                        height = safe_float_height(tags.get("height"), default_height)
                        levels = safe_float_height(tags.get("building:levels"), 0)
                        if levels != 0:
                            height = levels * 2.7
                        pending_nodes.append((height, x, y))
                        continue

                    if element["type"] == "relation":
                        outer_way = None
                        for member in element.get("members", []):
                            if (
                                member.get("type") == "way"
                                and member.get("role") == "outer"
                            ):
                                outer_way = ways_by_id.get(member["ref"])
                                if outer_way:
                                    break
                        if outer_way is None:
                            continue
                        node_ids = outer_way.get("nodes", [])
                        tags = element.get("tags") or outer_way.get("tags", {})
                    elif element["type"] == "way":
                        node_ids = element.get("nodes", [])
                        tags = element.get("tags", {})
                    else:
                        continue

                    footprint = []
                    for nid in node_ids:
                        if nid in node_xy:
                            x, y, _nlat, _nlon = node_xy[nid]
                            footprint.append((x, y))
                    if len(footprint) < 3:
                        continue

                    height = safe_float_height(tags.get("height"), default_height)
                    levels = safe_float_height(tags.get("building:levels"), 0)
                    if levels != 0:
                        height = levels * 2.7

                    z_offset = height * 0.002 * scaleHor * l_height_mult

                    poly = g2d.xy_ring_to_polygon(footprint)
                    if poly is None:
                        continue
                    if map_fp is not None:
                        poly = g2d.validate(poly.intersection(map_fp))
                    if poly is None or poly.is_empty:
                        continue

                    for part in g2d.iter_polygons(poly, min_area=min_area):
                        _puzzle_footprints.append((part, z_offset, "FLAT", 0.15))
                        _append_building(part, z_offset, _sample_z, b_verts, b_faces)

    # --- Node markers: cap by count (tallest first) and map containment ----
    if pending_nodes:
        if max_nodes > 0:
            pending_nodes.sort(key=lambda t: t[0], reverse=True)
            pending_nodes = pending_nodes[:max_nodes]
        for height, x, y in pending_nodes:
            if map_fp is not None and not map_fp.contains(Point(x, y)):
                continue
            z_offset = height * 0.002 * scaleHor * l_height_mult
            if z_offset < node_min_h_mm:
                z_offset = node_min_h_mm
            poly = Point(x, y).buffer(node_base_radius)
            _puzzle_footprints.append((poly, z_offset, "PYRAMID", 0.25))
            _append_node_landmark(
                x, y, z_offset, node_base_radius, _sample_z, b_verts, b_faces
            )

    if _ov.active:
        _ov.set_fetch_progress("landmarks", 0.75)
        _ov.update(message="Landmarks: building mesh…")
    remove_objects(wall_obj)

    global _puzzle_landmarks_data
    _puzzle_landmarks_data = (_puzzle_footprints, _sample_z)

    if not b_verts:
        return None

    mesh = bpy.data.meshes.new("landmark_mesh")
    mesh.from_pydata(b_verts, [], b_faces)
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new("Landmarks", mesh)
    bpy.context.collection.objects.link(obj)

    mesh.validate(verbose=False)
    mesh.update(calc_edges=True)
    bpy.context.view_layer.update()

    if _ov.active:
        _ov.set_fetch_progress("landmarks", 0.90)

    for poly in mesh.polygons:
        poly.use_smooth = False  # flat shading for landmarks

    recalculateNormals(obj)
    if _ov.active:
        _ov.set_fetch_progress("landmarks", 1.0)

    mat = bpy.data.materials.get("LANDMARKS")
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    print(f"[TP3D landmarks] final mesh build ({len(b_verts)} verts)")
    return obj
