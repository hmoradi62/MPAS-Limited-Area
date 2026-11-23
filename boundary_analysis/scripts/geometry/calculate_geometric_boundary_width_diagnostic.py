#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MPAS Limited-Area
Geometric Normal Boundary Width Analysis

Method:
1. Use midpoint of each inner boundary edge as ray origin.
2. Calculate local edge normal.
3. Determine outward direction using Mask 1 cell center.
4. Intersect the ray with all physical Mask 7 boundary edges.
5. Select the nearest valid positive intersection.
6. Compare geometric width with network-path width.

Outputs:
    geometric_boundary_width.csv
    geometric_boundary_width_report.txt
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from netCDF4 import Dataset
from pyproj import CRS, Transformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

NC_FILE = PROJECT_ROOT / "iran.static.nc"
INPUT_CSV = (
    PROJECT_ROOT
    / "boundary_analysis"
    / "results"
    / "geometric"
    / "local_boundary_width.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "boundary_analysis"
    / "results"
    / "geometric"
    / "diagnostic"
)
OUTPUT_CSV = OUTPUT_DIR / "geometric_boundary_width.csv"
REPORT_FILE = OUTPUT_DIR / "geometric_boundary_width_report.txt"

EXPECTED_INNER_EDGES = 207
EXPECTED_PHYSICAL_EDGES = 277

EARTH_RADIUS_KM = 6371.0088

# Numerical tolerances in projected meters
EPSILON = 1.0e-7
INTERSECTION_TOL = 1.0e-8

PROGRESS_INTERVAL = 25


# ============================================================
# BASIC GEOMETRY FUNCTIONS
# ============================================================

def cross2d(a, b):
    """2D cross product."""
    return a[0] * b[1] - a[1] * b[0]


def normalize(vector):
    """Return unit vector."""
    length = np.linalg.norm(vector)

    if length < EPSILON:
        raise ValueError("Zero-length vector encountered.")

    return vector / length


def convert_coordinates(values, units):
    """
    Convert NetCDF angular coordinates to degrees.

    Supports radians and degrees.
    """

    values = np.asarray(np.ma.filled(values, np.nan),
                        dtype=float)

    units = str(units).lower()

    if "radian" in units:
        return np.degrees(values)

    if "degree" in units:
        return values

    # Fallback for files without units attribute
    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        raise ValueError("Coordinate array contains no valid values.")

    if np.nanmax(np.abs(finite)) <= 2 * np.pi + 0.1:
        return np.degrees(values)

    return values


def make_local_projection(lon0, lat0):
    """
    Create local Azimuthal Equidistant projection.

    All distances are calculated in meters.
    """

    proj4 = (
        f"+proj=aeqd "
        f"+lat_0={lat0} "
        f"+lon_0={lon0} "
        f"+datum=WGS84 "
        f"+units=m "
        f"+no_defs"
    )

    local_crs = CRS.from_proj4(proj4)

    forward = Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True
    )

    inverse = Transformer.from_crs(
        local_crs,
        "EPSG:4326",
        always_xy=True
    )

    return forward, inverse


def ray_segment_intersection(origin, direction, a, b):
    """
    Intersection between:

        Ray:     P + t * direction, t >= 0
        Segment: A + u * (B-A), 0 <= u <= 1

    Returns:
        t, u, intersection point

    Returns None if no valid intersection exists.
    """

    segment = b - a

    denominator = cross2d(direction, segment)

    if abs(denominator) < EPSILON:
        return None

    delta = a - origin

    t = cross2d(delta, segment) / denominator
    u = cross2d(delta, direction) / denominator

    if t <= EPSILON:
        return None

    if u < -INTERSECTION_TOL:
        return None

    if u > 1.0 + INTERSECTION_TOL:
        return None

    intersection = origin + t * direction

    return float(t), float(u), intersection


# ============================================================
# LOAD MPAS DATA
# ============================================================

def load_mpas_data():

    if not os.path.exists(NC_FILE):
        raise FileNotFoundError(NC_FILE)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(INPUT_CSV)

    print("\nLoading MPAS static file...")

    with Dataset(NC_FILE, "r") as nc:

        lat_cell = convert_coordinates(
            nc.variables["latCell"][:],
            nc.variables["latCell"].getncattr("units")
            if "units" in nc.variables["latCell"].ncattrs()
            else ""
        )

        lon_cell = convert_coordinates(
            nc.variables["lonCell"][:],
            nc.variables["lonCell"].getncattr("units")
            if "units" in nc.variables["lonCell"].ncattrs()
            else ""
        )

        lat_vertex = convert_coordinates(
            nc.variables["latVertex"][:],
            nc.variables["latVertex"].getncattr("units")
            if "units" in nc.variables["latVertex"].ncattrs()
            else ""
        )

        lon_vertex = convert_coordinates(
            nc.variables["lonVertex"][:],
            nc.variables["lonVertex"].getncattr("units")
            if "units" in nc.variables["lonVertex"].ncattrs()
            else ""
        )

        cell_mask = np.asarray(
            np.ma.filled(
                nc.variables["bdyMaskCell"][:], -999
            ),
            dtype=int
        )

        cells_on_edge = np.asarray(
            np.ma.filled(
                nc.variables["cellsOnEdge"][:], 0
            ),
            dtype=int
        )

        vertices_on_edge = np.asarray(
            np.ma.filled(
                nc.variables["verticesOnEdge"][:], 0
            ),
            dtype=int
        )

    print("MPAS data loaded.")
    print("Cells:", len(lat_cell))
    print("Edges:", len(cells_on_edge))
    print("Vertices:", len(lat_vertex))

    return {
        "lat_cell": lat_cell,
        "lon_cell": lon_cell,
        "lat_vertex": lat_vertex,
        "lon_vertex": lon_vertex,
        "cell_mask": cell_mask,
        "cells_on_edge": cells_on_edge,
        "vertices_on_edge": vertices_on_edge
    }


# ============================================================
# IDENTIFY PHYSICAL BOUNDARY
# ============================================================

def identify_physical_boundary(data):

    cell_mask = data["cell_mask"]
    cells_on_edge = data["cells_on_edge"]

    physical_edges = []

    for edge_id, cells in enumerate(cells_on_edge):

        c1, c2 = map(int, cells)

        # Exactly one missing cell
        if (c1 == 0) == (c2 == 0):
            continue

        valid_cell = c2 if c1 == 0 else c1

        if not (1 <= valid_cell <= len(cell_mask)):
            continue

        if cell_mask[valid_cell - 1] == 7:
            physical_edges.append(edge_id)

    print("\nPhysical boundary detection:")
    print("Physical Mask 7 edges:", len(physical_edges))

    if len(physical_edges) != EXPECTED_PHYSICAL_EDGES:
        raise ValueError(
            f"Expected {EXPECTED_PHYSICAL_EDGES} physical edges, "
            f"found {len(physical_edges)}."
        )

    return physical_edges

# ============================================================
# GEOMETRIC WIDTH FOR ONE INNER EDGE
# ============================================================

def calculate_one_edge(row, data, physical_edges):

    edge_id = int(row["inner_edge"])
    start_cell = int(row["start_cell"])

    network_width = float(row["width_km"])
    network_outer_edge = int(row["outer_edge"])

    lat_cell = data["lat_cell"]
    lon_cell = data["lon_cell"]

    lat_vertex = data["lat_vertex"]
    lon_vertex = data["lon_vertex"]

    cell_mask = data["cell_mask"]

    cells_on_edge = data["cells_on_edge"]
    vertices_on_edge = data["vertices_on_edge"]

    n_cells = len(cell_mask)
    n_edges = len(cells_on_edge)

    # --------------------------------------------------------
    # Validate IDs (CSV IDs are zero-based)
    # --------------------------------------------------------

    edge_idx = edge_id
    cell_idx = start_cell

    if not (0 <= edge_idx < n_edges):
        raise ValueError(f"Invalid inner edge: {edge_id}")

    if not (0 <= cell_idx < n_cells):
        raise ValueError(f"Invalid start cell: {start_cell}")

    if cell_mask[cell_idx] != 1:
        raise ValueError(
            f"Start cell {start_cell} is not Mask 1."
        )

    adjacent_cells = cells_on_edge[edge_idx]

    adjacent_cells_zero = [
        int(c) - 1
        for c in adjacent_cells
        if int(c) > 0
    ]

    if start_cell not in adjacent_cells_zero:
        raise ValueError(
            f"Start cell {start_cell} is not adjacent "
            f"to edge {edge_id}."
        )

    # --------------------------------------------------------
    # Read complete Dijkstra path
    # --------------------------------------------------------

    path_cell_ids = [
        int(x)
        for x in str(row["path_cell_ids"]).split(";")
        if str(x).strip()
    ]

    if not path_cell_ids:
        raise ValueError(
            f"Empty path_cell_ids for edge {edge_id}."
        )

    outer_cell = path_cell_ids[-1]

    if not (0 <= outer_cell < n_cells):
        raise ValueError(
            f"Invalid outer cell {outer_cell}."
        )

    if cell_mask[outer_cell] != 7:
        raise ValueError(
            f"Outer cell {outer_cell} is not Mask 7."
        )

    # --------------------------------------------------------
    # Validate network outer edge
    # --------------------------------------------------------

    if not (0 <= network_outer_edge < n_edges):
        raise ValueError(
            f"Invalid network outer edge: {network_outer_edge}"
        )

    if network_outer_edge not in physical_edges:
        raise ValueError(
            f"Network outer edge {network_outer_edge} "
            f"is not a physical boundary edge."
        )

    network_edge_cells = [
        int(c) - 1
        for c in cells_on_edge[network_outer_edge]
        if int(c) > 0
    ]

    if outer_cell not in network_edge_cells:
        raise ValueError(
            f"Network outer edge {network_outer_edge} "
            f"is not adjacent to outer cell {outer_cell}."
        )

    # --------------------------------------------------------
    # Build cell -> physical boundary edges
    # --------------------------------------------------------

    boundary_edges_by_cell = {}

    for outer_id in physical_edges:

        cells = cells_on_edge[outer_id]

        for c in cells:

            c = int(c)

            if c > 0:

                cell_id = c - 1

                boundary_edges_by_cell.setdefault(
                    cell_id, []
                ).append(outer_id)

    # --------------------------------------------------------
    # Build cell adjacency from cellsOnEdge
    # --------------------------------------------------------

    neighbors = [set() for _ in range(n_cells)]

    for cells in cells_on_edge:

        c1, c2 = map(int, cells)

        if c1 > 0 and c2 > 0:

            i = c1 - 1
            j = c2 - 1

            neighbors[i].add(j)
            neighbors[j].add(i)

    # --------------------------------------------------------
    # Construct local boundary candidates
    # --------------------------------------------------------

    frontier = {outer_cell}
    visited = {outer_cell}

    candidate_levels = []

    for ring in range(3):

        ring_edges = set()

        for cell_id in frontier:

            ring_edges.update(
                boundary_edges_by_cell.get(cell_id, [])
            )

        if ring_edges:

            candidate_levels.append({
                "ring": ring,
                "edges": sorted(ring_edges)
            })

        next_frontier = set()

        for cell_id in frontier:

            for neighbor_id in neighbors[cell_id]:

                if neighbor_id not in visited:

                    visited.add(neighbor_id)
                    next_frontier.add(neighbor_id)

        frontier = next_frontier

        if not frontier:
            break

    # --------------------------------------------------------
    # Get vertices of inner edge
    # --------------------------------------------------------

    vertex_ids = vertices_on_edge[edge_idx]

    if np.any(vertex_ids == 0):
        raise ValueError(
            f"Invalid vertices for inner edge {edge_id}."
        )

    v1 = int(vertex_ids[0]) - 1
    v2 = int(vertex_ids[1]) - 1

    lon1 = lon_vertex[v1]
    lat1 = lat_vertex[v1]

    lon2 = lon_vertex[v2]
    lat2 = lat_vertex[v2]

    # --------------------------------------------------------
    # Local projection
    # --------------------------------------------------------

    lon0 = np.degrees(
        np.arctan2(
            np.sin(np.radians(lon1)) +
            np.sin(np.radians(lon2)),

            np.cos(np.radians(lon1)) +
            np.cos(np.radians(lon2))
        )
    )

    lat0 = (lat1 + lat2) / 2.0

    forward, inverse = make_local_projection(lon0, lat0)

    x1, y1 = forward.transform(lon1, lat1)
    x2, y2 = forward.transform(lon2, lat2)

    p1 = np.array([x1, y1], dtype=float)
    p2 = np.array([x2, y2], dtype=float)

    origin = (p1 + p2) / 2.0

    # --------------------------------------------------------
    # Tangent and normal
    # --------------------------------------------------------

    tangent = normalize(p2 - p1)

    normal = np.array(
        [-tangent[1], tangent[0]],
        dtype=float
    )

    # --------------------------------------------------------
    # --------------------------------------------------------
    # Determine boundary-zone direction
    # Direction must point from Mask 0/1 interface
    # toward Mask 1 and subsequently Mask 7.
    # --------------------------------------------------------

    cx, cy = forward.transform(
        lon_cell[cell_idx],
        lat_cell[cell_idx]
    )

    cell_center = np.array([cx, cy], dtype=float)

    inward_vector = cell_center - origin

    dot_product = np.dot(normal, inward_vector)

    # Make normal point toward Mask 1
    if dot_product < 0:
        normal = -normal

    # Validate direction
    if np.dot(normal, inward_vector) <= 0:
        raise ValueError(
          f"Cannot determine boundary-zone direction "
          f"for edge {edge_id}."
    )
    # DIAGNOSTIC: Compare normal direction with network path
    # --------------------------------------------------------
    # DIAGNOSTIC: Trace ray through all MPAS mask layers
    diagnostic_edges = {769, 777, 828}

    # Full topology diagnostic is executed after the algorithm selects
    # its physical boundary edge, so both results can be compared.

    if edge_id in diagnostic_edges:

        print("\n" + "=" * 65)
        print(f"GEOMETRIC DIRECTION DIAGNOSTIC - EDGE {edge_id}")
        print("=" * 65)

        # Transform network outer edge vertices
        net_vertices = vertices_on_edge[network_outer_edge]

        nv1, nv2 = map(int, net_vertices)

        nv1 -= 1
        nv2 -= 1

        nax, nay = forward.transform(
            lon_vertex[nv1],
            lat_vertex[nv1]
        )

        nbx, nby = forward.transform(
            lon_vertex[nv2],
            lat_vertex[nv2]
        )

        A = np.array([nax, nay])
        B = np.array([nbx, nby])

        # Signed coordinates relative to normal ray
        qA = np.dot(A - origin, tangent)
        qB = np.dot(B - origin, tangent)

        sA = np.dot(A - origin, normal)
        sB = np.dot(B - origin, normal)

        print("Network outer edge:", network_outer_edge)
        print("Outer cell:", outer_cell)

        print("\nNetwork edge endpoint coordinates:")
        print(f"Endpoint A: along={sA/1000:.2f} km, cross={qA/1000:.2f} km")
        print(f"Endpoint B: along={sB/1000:.2f} km, cross={qB/1000:.2f} km")

        # Outer cell center relative to normal ray
        ocx, ocy = forward.transform(
            lon_cell[outer_cell],
            lat_cell[outer_cell]
        )

        outer_center = np.array([ocx, ocy])

        along_outer = np.dot(
            outer_center - origin,
            normal
        )

        cross_outer = np.dot(
            outer_center - origin,
            tangent
        )

        print("\nOuter cell center:")
        print(f"Along normal: {along_outer/1000:.2f} km")
        print(f"Cross-track: {cross_outer/1000:.2f} km")

        # Test direct intersection with network outer edge
        net_intersection = ray_segment_intersection(
            origin,
            normal,
            A,
            B
        )

        print("\nRay intersection with network outer edge:")

        if net_intersection is None:
            print("NO INTERSECTION")
        else:
            distance, fraction, point = net_intersection
            print(f"Intersection distance: {distance/1000:.2f} km")
            print(f"Segment fraction: {fraction:.4f}")

        print("=" * 65)

    # --------------------------------------------------------
        # --------------------------------------------------------
    # GLOBAL PHYSICAL BOUNDARY SEARCH
    # Diagnostic test: check all physical boundary edges
    # --------------------------------------------------------

    candidate_levels = [{
        "ring": -1,
        "edges": sorted(physical_edges)
    }]

    nearest = None
    selected_ring = None
    tested_edges = 0

    for level in candidate_levels:

        ring = level["ring"]
        candidate_edges = level["edges"]

        intersections = []

        for outer_id in candidate_edges:

            tested_edges += 1

            outer_vertices = vertices_on_edge[outer_id]

            ov1, ov2 = map(int, outer_vertices)

            if ov1 == 0 or ov2 == 0:
                continue

            ov1 -= 1
            ov2 -= 1

            ax, ay = forward.transform(
                lon_vertex[ov1],
                lat_vertex[ov1]
            )

            bx, by = forward.transform(
                lon_vertex[ov2],
                lat_vertex[ov2]
            )

            a = np.array([ax, ay], dtype=float)
            b = np.array([bx, by], dtype=float)

            result = ray_segment_intersection(
                origin,
                normal,
                a,
                b
            )

            if result is None:
                continue

            distance_m, fraction, point = result

            intersections.append({
                "outer_edge": outer_id,
                "distance_m": distance_m,
                "fraction": fraction,
                "point": point
            })

        if intersections:

            nearest = min(
                intersections,
                key=lambda item: item["distance_m"]
            )

            selected_ring = ring
            break

    # --------------------------------------------------------
    # Diagnostic only: trace the ray through the actual MPAS mesh.
    # This does not modify the selected width or its search logic.
    # --------------------------------------------------------
    if edge_id in diagnostic_edges and nearest is not None:
        diagnose_ray_mesh_crossings(
            edge_id=edge_id,
            start_cell=start_cell,
            origin=origin,
            direction=normal,
            vertices_on_edge=vertices_on_edge,
            cells_on_edge=cells_on_edge,
            lat_vertex=lat_vertex,
            lon_vertex=lon_vertex,
            bdy_mask_cell=cell_mask,
            physical_edges=physical_edges,
            selected_edge=nearest["outer_edge"],
            selected_distance_km=nearest["distance_m"] / 1000.0,
            forward=forward,
            vertex_tolerance_m=0.05
        )

    # --------------------------------------------------------
    # No local intersection
    # --------------------------------------------------------

    if nearest is None:

        return {
            "inner_edge": edge_id,
            "start_cell": start_cell,

            "network_outer_edge": network_outer_edge,
            "network_width_km": network_width,

            "normal_outer_edge": np.nan,
            "normal_width_km": np.nan,

            "difference_km": np.nan,
            "difference_percent": np.nan,

            "intersection_lon": np.nan,
            "intersection_lat": np.nan,
            "intersection_fraction": np.nan,

            "normal_x": normal[0],
            "normal_y": normal[1],

            "outer_cell": outer_cell,
            "search_ring": np.nan,
            "candidate_edges_tested": tested_edges,

            "status": "NO_LOCAL_INTERSECTION"
        }
    # --------------------------------------------------------
    # DIAGNOSTIC: Selected global intersection edge
    # --------------------------------------------------------

    if edge_id in diagnostic_edges:

        selected_edge = nearest["outer_edge"]

        selected_vertices = vertices_on_edge[selected_edge]

        sv1, sv2 = map(int, selected_vertices)

        sv1 -= 1
        sv2 -= 1

        sax, say = forward.transform(
            lon_vertex[sv1],
            lat_vertex[sv1]
        )

        sbx, sby = forward.transform(
            lon_vertex[sv2],
            lat_vertex[sv2]
        )

        SA = np.array([sax, say])
        SB = np.array([sbx, sby])

        # Coordinates relative to normal and tangent
        sa_normal = np.dot(SA - origin, normal)
        sb_normal = np.dot(SB - origin, normal)

        sa_cross = np.dot(SA - origin, tangent)
        sb_cross = np.dot(SB - origin, tangent)

        print("\nSELECTED GLOBAL BOUNDARY EDGE")
        print("Inner edge:", edge_id)
        print("Selected physical edge:", selected_edge)

        print(
            "Intersection distance:",
            nearest["distance_m"] / 1000,
            "km"
        )

        print(
            "Endpoint A: along=%.2f km, cross=%.2f km"
            % (sa_normal / 1000, sa_cross / 1000)
        )

        print(
            "Endpoint B: along=%.2f km, cross=%.2f km"
            % (sb_normal / 1000, sb_cross / 1000)
        )

        print(
            "Intersection fraction:",
            nearest["fraction"]
        )

        print(
            "Intersection coordinate:",
            inverse.transform(
                nearest["point"][0],
                nearest["point"][1]
            )
        )
    # --------------------------------------------------------
    # Calculate normal width
    # --------------------------------------------------------

    distance_km = nearest["distance_m"] / 1000.0

    hit_x, hit_y = nearest["point"]

    hit_lon, hit_lat = inverse.transform(
        hit_x,
        hit_y
    )

    difference = distance_km - network_width

    difference_percent = (
        100.0 * difference / network_width
        if network_width > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "inner_edge": edge_id,
        "start_cell": start_cell,

        "network_outer_edge": network_outer_edge,
        "network_width_km": network_width,

        "normal_outer_edge": nearest["outer_edge"],
        "normal_width_km": distance_km,

        "difference_km": difference,
        "difference_percent": difference_percent,

        "intersection_lon": hit_lon,
        "intersection_lat": hit_lat,
        "intersection_fraction": nearest["fraction"],

        "normal_x": normal[0],
        "normal_y": normal[1],

        "outer_cell": outer_cell,
        "search_ring": selected_ring,
        "candidate_edges_tested": tested_edges,

        "status": "SUCCESS"
    }

# ============================================================


def diagnose_ray_mesh_crossings(
    edge_id,
    start_cell,
    origin,
    direction,
    vertices_on_edge,
    cells_on_edge,
    lat_vertex,
    lon_vertex,
    bdy_mask_cell,
    physical_edges,
    selected_edge,
    selected_distance_km,
    forward,
    vertex_tolerance_m=0.05
):
    """Trace all ray/edge intersections and test cell-to-cell continuity.

    This is diagnostic-only. It does not choose or alter the geometric width.
    MPAS connectivity in cellsOnEdge and verticesOnEdge is one-based, with
    zero denoting a missing cell/vertex.
    """

    physical_edge_set = {int(e) for e in physical_edges}
    crossings = []

    for mesh_edge in range(len(vertices_on_edge)):
        v1, v2 = map(int, vertices_on_edge[mesh_edge])
        if v1 <= 0 or v2 <= 0:
            continue
        v1 -= 1
        v2 -= 1

        ax, ay = forward.transform(lon_vertex[v1], lat_vertex[v1])
        bx, by = forward.transform(lon_vertex[v2], lat_vertex[v2])
        A = np.array([ax, ay], dtype=float)
        B = np.array([bx, by], dtype=float)

        hit = ray_segment_intersection(origin, direction, A, B)
        if hit is None:
            continue

        distance_m, fraction, point = hit
        c1_raw, c2_raw = map(int, cells_on_edge[mesh_edge])
        c1 = c1_raw - 1 if c1_raw > 0 else -1
        c2 = c2_raw - 1 if c2_raw > 0 else -1
        m1 = int(bdy_mask_cell[c1]) if 0 <= c1 < len(bdy_mask_cell) else -1
        m2 = int(bdy_mask_cell[c2]) if 0 <= c2 < len(bdy_mask_cell) else -1

        crossings.append({
            "distance_m": float(distance_m),
            "distance_km": float(distance_m / 1000.0),
            "edge": int(mesh_edge),
            "cell1": int(c1),
            "cell2": int(c2),
            "mask1": m1,
            "mask2": m2,
            "fraction": float(fraction),
            "physical": int(mesh_edge) in physical_edge_set
        })

    crossings.sort(key=lambda item: (item["distance_m"], item["edge"]))

    print("\n" + "=" * 100)
    print(f"RAY / MPAS TOPOLOGY DIAGNOSTIC - INNER EDGE {edge_id}")
    print("=" * 100)
    print(f"Starting cell: {start_cell} | mask: {int(bdy_mask_cell[start_cell])}")
    print(f"Algorithm-selected physical edge: {selected_edge}")
    print(f"Algorithm-selected distance: {selected_distance_km:.3f} km")
    print(f"Total mesh-edge intersections: {len(crossings)}")

    if not crossings:
        print("No mesh-edge intersections were found.")
        print("=" * 100)
        return

    physical_hits = [item for item in crossings if item["physical"]]
    print("First geometric mesh intersection:", f"{crossings[0]['distance_km']:.3f} km",
          f"(edge {crossings[0]['edge']})")
    if physical_hits:
        first_physical = physical_hits[0]
        print("First physical-boundary intersection:",
              f"{first_physical['distance_km']:.3f} km",
              f"(edge {first_physical['edge']})")
        print("Algorithm selected first physical hit:",
              first_physical["edge"] == int(selected_edge))
    else:
        first_physical = None
        print("No physical-boundary edge intersects this ray.")

    print("\nALL INTERSECTIONS (physical edges marked with '*')")
    print(f"{'No.':<5}{'Dist(km)':<12}{'Edge':<8}{'Cell 1':<9}{'Mask 1':<8}"
          f"{'Cell 2':<9}{'Mask 2':<8}{'Frac':<9}{'Type':<10}")
    for i, item in enumerate(crossings, 1):
        kind = "PHYSICAL*" if item["physical"] else "internal"
        print(f"{i:<5}{item['distance_km']:<12.3f}{item['edge']:<8}"
              f"{item['cell1']:<9}{item['mask1']:<8}"
              f"{item['cell2']:<9}{item['mask2']:<8}"
              f"{item['fraction']:<9.4f}{kind:<10}")

    # Walk the edge intersections in distance order. Only an edge touching
    # the current cell is a valid next transition in the cell topology.
    current_cell = int(start_cell)
    visited_cells = [current_cell]
    mask_sequence = [int(bdy_mask_cell[current_cell])]
    connected_edges = []
    connected_distance = None
    stop_reason = "Ray ended before a physical boundary."
    reached_physical = False

    i = 0
    while i < len(crossings):
        group = [crossings[i]]
        j = i + 1
        while j < len(crossings) and abs(
            crossings[j]["distance_m"] - crossings[i]["distance_m"]
        ) <= vertex_tolerance_m:
            group.append(crossings[j])
            j += 1

        touching = [x for x in group if current_cell in (x["cell1"], x["cell2"])]
        if len(touching) == 0:
            stop_reason = (
                f"Disconnected intersection at {group[0]['distance_km']:.3f} km; "
                f"none of the edges in this crossing group touches current cell {current_cell}."
            )
            break
        if len(touching) > 1:
            edge_ids = [x["edge"] for x in touching]
            stop_reason = (
                f"Ambiguous vertex hit at {group[0]['distance_km']:.3f} km; "
                f"multiple intersected edges {edge_ids} touch current cell {current_cell}. "
                "No arbitrary transition was selected."
            )
            break

        item = touching[0]
        connected_edges.append(item["edge"])
        connected_distance = item["distance_km"]
        other_cell = item["cell2"] if item["cell1"] == current_cell else item["cell1"]

        if other_cell < 0:
            if item["physical"] and int(bdy_mask_cell[current_cell]) == 7:
                reached_physical = True
                stop_reason = "Connected ray exits the domain from a Mask 7 cell."
            else:
                stop_reason = (
                    f"Ray reaches a domain edge {item['edge']} from cell {current_cell}, "
                    "but it is not classified as a Mask 7 physical boundary."
                )
            break

        current_cell = int(other_cell)
        visited_cells.append(current_cell)
        mask_sequence.append(int(bdy_mask_cell[current_cell]))
        i = j

    # Compress repeated adjacent masks, preserving the exact cell sequence above.
    compressed_masks = []
    for value in mask_sequence:
        if not compressed_masks or value != compressed_masks[-1]:
            compressed_masks.append(value)

    print("\nCONNECTED CELL-TO-CELL TRACE")
    print("Visited cells:", " -> ".join(map(str, visited_cells)))
    print("Mask sequence:", " -> ".join(map(str, mask_sequence)))
    print("Compressed mask sequence:", " -> ".join(map(str, compressed_masks)))
    print("Connected mesh edges:", " -> ".join(map(str, connected_edges)) if connected_edges else "None")
    print("Trace status:", stop_reason)
    print("Reached physical boundary through connected topology:", reached_physical)
    if connected_distance is not None:
        print(f"Connected trace terminal distance: {connected_distance:.3f} km")

    if first_physical is not None:
        print("\nCOMPARISON")
        print(f"First physical edge on geometric ray: {first_physical['edge']}")
        print(f"Algorithm-selected edge: {selected_edge}")
        print(f"Distance difference: {selected_distance_km - first_physical['distance_km']:+.3f} km")
        print("Selected edge equals first physical ray intersection:",
              int(selected_edge) == first_physical["edge"])

    print("=" * 100)


# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MPAS GEOMETRIC BOUNDARY WIDTH ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_CSV)

    print("\nInner interface edges:", len(df))

    if len(df) != EXPECTED_INNER_EDGES:
        raise ValueError(
            f"Expected {EXPECTED_INNER_EDGES} inner edges, "
            f"found {len(df)}."
        )

    # --------------------------------------------------------
    # Load NetCDF
    # --------------------------------------------------------

    data = load_mpas_data()

    # --------------------------------------------------------
    # Identify physical boundary
    # --------------------------------------------------------

    physical_edges = identify_physical_boundary(data)

    # --------------------------------------------------------
    # Calculate widths
    # --------------------------------------------------------

    results = []

    failures = []

    print("\nCalculating geometric widths...")

    for i, (_, row) in enumerate(df.iterrows(), start=1):

        try:

            result = calculate_one_edge(
                row,
                data,
                physical_edges
            )

            results.append(result)

            if result["status"] != "SUCCESS":
                failures.append(result["inner_edge"])

        except Exception as exc:

            failures.append(int(row["inner_edge"]))

            results.append({
                 "inner_edge": int(row["inner_edge"]),
                 "start_cell": int(row["start_cell"]),

                 "network_outer_edge": int(row["outer_edge"]),
                 "network_width_km": float(row["width_km"]),

                 "normal_outer_edge": np.nan,
                 "normal_width_km": np.nan,

                 "difference_km": np.nan,
                 "difference_percent": np.nan,

                 "intersection_lon": np.nan,
                 "intersection_lat": np.nan,
                 "intersection_fraction": np.nan,

                 "outer_cell": np.nan,
                 "search_ring": np.nan,
                 "candidate_edges_tested": 0,

                 "normal_x": np.nan,
                 "normal_y": np.nan,

                 "status": f"ERROR: {str(exc)}"
})
            warnings.warn(
                f"Edge {int(row['inner_edge'])}: {exc}"
            )

        if i % PROGRESS_INTERVAL == 0 or i == len(df):
            print(f"Processed {i}/{len(df)}")

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output = pd.DataFrame(results)

    output.to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.6f"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    successful = output[
        output["status"] == "SUCCESS"
    ]

    widths = successful["normal_width_km"].dropna()

    print("\n" + "=" * 70)
    print("GEOMETRIC WIDTH STATISTICS")
    print("=" * 70)

    print("Total inner edges:", len(df))
    print("Successful paths:", len(successful))
    print("Failed paths:", len(df) - len(successful))

    if len(widths) > 0:

        print(f"Minimum : {widths.min():.2f} km")
        print(f"P05     : {widths.quantile(0.05):.2f} km")
        print(f"P25     : {widths.quantile(0.25):.2f} km")
        print(f"Median  : {widths.median():.2f} km")
        print(f"Mean    : {widths.mean():.2f} km")
        print(f"P75     : {widths.quantile(0.75):.2f} km")
        print(f"P95     : {widths.quantile(0.95):.2f} km")
        print(f"Maximum : {widths.max():.2f} km")

    # --------------------------------------------------------
    # Comparison statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NETWORK VS GEOMETRIC WIDTH")
    print("=" * 70)

    if len(successful) > 0:

        diff = successful["difference_km"]
        pct = successful["difference_percent"]

        print(f"Mean difference : {diff.mean():.3f} km")
        print(f"Mean absolute difference : {diff.abs().mean():.3f} km")
        print(f"Maximum absolute difference : {diff.abs().max():.3f} km")

        print(f"Mean percentage difference : {pct.mean():.2f}%")
        print(f"Mean absolute percentage difference : {pct.abs().mean():.2f}%")

        print(
            "Geometric width greater than network width:",
            int((diff > 0).sum())
        )

        print(
            "Geometric width smaller than network width:",
            int((diff < 0).sum())
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    with open(REPORT_FILE, "w", encoding="utf-8") as f:

        f.write("MPAS GEOMETRIC BOUNDARY WIDTH ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Inner edges: {len(df)}\n")
        f.write(f"Physical boundary edges: {len(physical_edges)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(df) - len(successful)}\n\n")

        if len(widths) > 0:

            f.write("GEOMETRIC WIDTH STATISTICS\n")
            f.write("-" * 40 + "\n")

            f.write(f"Minimum: {widths.min():.4f} km\n")
            f.write(f"Median: {widths.median():.4f} km\n")
            f.write(f"Mean: {widths.mean():.4f} km\n")
            f.write(f"Maximum: {widths.max():.4f} km\n\n")

        if len(successful) > 0:

            f.write("NETWORK COMPARISON\n")
            f.write("-" * 40 + "\n")

            f.write(
                f"Mean absolute difference: "
                f"{successful['difference_km'].abs().mean():.4f} km\n"
            )

            f.write(
                f"Mean absolute percentage difference: "
                f"{successful['difference_percent'].abs().mean():.2f}%\n"
            )

        if failures:

            f.write("\nFAILED EDGES\n")
            f.write(str(failures))

    print("\nSaved:", OUTPUT_CSV)
    print("Saved:", REPORT_FILE)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()
