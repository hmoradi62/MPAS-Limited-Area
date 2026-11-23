#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Topology-aware tracing of geometric normal rays through an MPAS mesh.

Inputs (in the current working directory by default):
    iran.static.nc
    local_boundary_width.csv

The normal construction is intentionally consistent with the comparison
workflow: start at the inner-edge midpoint, build a local AEQD projection,
construct the edge normal, and orient it toward the CSV start cell (Mask 1).

Unlike a global geometric ray/physical-boundary intersection, this script
walks only through edges incident to the current cell, using cellsOnEdge.

Outputs:
    boundary_ray_topology.csv
    boundary_ray_topology_report.txt

Dependencies:
    numpy, pandas, netCDF4, pyproj
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import CRS, Transformer


EXPECTED_MASK_SEQUENCE = [1, 2, 3, 4, 5, 6, 7]
EARTH_RADIUS_KM = 6371.0

# Geometric tolerances in the local projected coordinate system.
INTERSECTION_TOL_M = 0.05
SEGMENT_TOL = 1.0e-9
VERTEX_TIE_TOL_M = 0.25
MAX_STEPS = 100


def first_existing(ds, names):
    for name in names:
        if name in ds.variables:
            return ds.variables[name][:]
    raise KeyError(f"None of these variables exists in NetCDF: {names}")


def as_float_array(var):
    return np.asarray(np.ma.filled(var, np.nan), dtype=float)


def unit_vector(v):
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Cannot normalize a zero or invalid vector.")
    return v / norm


def cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def ray_segment_intersection(origin, direction, p1, p2):
    """
    Intersect origin + t*direction (t >= 0) with p1 + u*(p2-p1), 0<=u<=1.
    direction must be unit length, so t is in metres.
    Returns (t, u) or None.
    """
    segment = p2 - p1
    denom = cross2(direction, segment)

    if abs(denom) < 1.0e-12:
        return None

    delta = p1 - origin
    t = cross2(delta, segment) / denom
    u = cross2(delta, direction) / denom

    if t < -INTERSECTION_TOL_M:
        return None
    if u < -SEGMENT_TOL or u > 1.0 + SEGMENT_TOL:
        return None

    return max(0.0, float(t)), float(min(1.0, max(0.0, u)))


def make_transformer(lon0, lat0):
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.12f} +lon_0={lon0:.12f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)


def edge_midpoint_lonlat(edge_id, vertices_on_edge, lat_vertex, lon_vertex):
    # edge_id is zero-based; verticesOnEdge is one-based.
    v1, v2 = vertices_on_edge[edge_id]
    if v1 <= 0 or v2 <= 0:
        raise ValueError(f"Edge {edge_id} has missing endpoint vertex.")
    i1, i2 = int(v1) - 1, int(v2) - 1

    # Midpoint is calculated in 3-D unit-sphere coordinates to handle
    # longitude wraparound more safely than arithmetic lon averaging.
    def xyz(lat, lon):
        latr, lonr = np.deg2rad(lat), np.deg2rad(lon)
        return np.array([
            np.cos(latr) * np.cos(lonr),
            np.cos(latr) * np.sin(lonr),
            np.sin(latr),
        ])

    m = unit_vector(xyz(lat_vertex[i1], lon_vertex[i1]) +
                    xyz(lat_vertex[i2], lon_vertex[i2]))
    lat = math.degrees(math.asin(m[2]))
    lon = math.degrees(math.atan2(m[1], m[0]))
    return lon, lat


def build_edge_cell_adjacency(cells_on_edge, n_cells):
    cell_edges = [[] for _ in range(n_cells)]
    for e, pair in enumerate(cells_on_edge):
        for c1 in pair:
            c = int(c1) - 1
            if c >= 0:
                if c >= n_cells:
                    raise ValueError(
                        f"cellsOnEdge contains out-of-range cell {c1} at edge {e}"
                    )
                cell_edges[c].append(e)
    return cell_edges


def project_mesh(transformer, lon_vertex, lat_vertex, vertices_on_edge):
    x_vertex, y_vertex = transformer.transform(lon_vertex, lat_vertex)
    xy_vertex = np.column_stack([x_vertex, y_vertex])

    n_edges = vertices_on_edge.shape[0]
    edge_xy = np.full((n_edges, 2, 2), np.nan, dtype=float)

    for e, pair in enumerate(vertices_on_edge):
        v1, v2 = map(int, pair)
        if v1 <= 0 or v2 <= 0:
            continue
        edge_xy[e, 0] = xy_vertex[v1 - 1]
        edge_xy[e, 1] = xy_vertex[v2 - 1]

    return edge_xy


def trace_one_ray(
    inner_edge,
    start_cell,
    network_outer_edge,
    cells_on_edge,
    bdy_mask_cell,
    cell_edges,
    edge_xy,
    origin,
    direction,
    global_nearest_edge,
    global_nearest_distance_km,
):
    n_cells = len(bdy_mask_cell)
    current_cell = int(start_cell)
    if current_cell < 0 or current_cell >= n_cells:
        return {
            "status": "INVALID_START_CELL",
            "error": f"start_cell index {start_cell} is out of range",
        }

    start_mask = int(bdy_mask_cell[current_cell])
    if start_mask != 1:
        return {
            "status": "INVALID_START_MASK",
            "error": f"start cell mask is {start_mask}, expected 1",
        }

    visited_cells = [current_cell]
    visited_edges = []
    mask_sequence = [start_mask]
    current_t = 0.0
    exit_edge = -1
    exit_cell = -1
    exit_distance_km = np.nan
    exit_x = np.nan
    exit_y = np.nan
    status = None
    error = ""

    for step in range(MAX_STEPS):
        candidates = []

        for e in cell_edges[current_cell]:
            if e in visited_edges:
                continue

            p1, p2 = edge_xy[e]
            if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
                continue

            hit = ray_segment_intersection(origin, direction, p1, p2)
            if hit is None:
                continue

            t, u = hit
            # Ignore the entry edge / current starting interface and any
            # intersections not strictly ahead of the current position.
            if t <= current_t + INTERSECTION_TOL_M:
                continue

            cpair = [int(v) - 1 for v in cells_on_edge[e]]
            adjacent = [c for c in cpair if c >= 0 and c != current_cell]
            if len(adjacent) > 1:
                candidates.append((t, e, adjacent, u, "BAD_CONNECTIVITY"))
            else:
                candidates.append((t, e, adjacent, u, "OK"))

        if not candidates:
            status = "NO_HIT"
            error = "No forward intersection found among edges of current cell"
            break

        candidates.sort(key=lambda item: item[0])
        t_min = candidates[0][0]
        tied = [c for c in candidates if abs(c[0] - t_min) <= VERTEX_TIE_TOL_M]

        if len(tied) > 1:
            status = "VERTEX_AMBIGUITY"
            error = (
                "Ray intersects multiple incident edges at nearly the same "
                f"distance ({t_min / 1000.0:.6f} km); branch not guessed"
            )
            break

        t, e, adjacent, u, connectivity_status = tied[0]

        if connectivity_status != "OK":
            status = "INVALID_CONNECTIVITY"
            error = f"Edge {e} has invalid cellsOnEdge connectivity"
            break

        visited_edges.append(int(e))
        hit_xy = origin + t * direction

        # A boundary edge has exactly one adjacent cell in cellsOnEdge.
        if len(adjacent) == 0:
            current_mask = int(bdy_mask_cell[current_cell])
            if current_mask == 7:
                status = "BOUNDARY_EXIT"
                exit_edge = int(e)
                exit_cell = int(current_cell)
                exit_distance_km = t / 1000.0
                exit_x, exit_y = float(hit_xy[0]), float(hit_xy[1])
            else:
                status = "INVALID_EARLY_BOUNDARY"
                error = (
                    f"Ray reached domain boundary at edge {e} while current "
                    f"cell {current_cell} has mask {current_mask}, not 7"
                )
            break

        next_cell = int(adjacent[0])
        if next_cell in visited_cells:
            status = "INVALID_LOOP"
            error = f"Ray revisited cell {next_cell}"
            break

        next_mask = int(bdy_mask_cell[next_cell])
        visited_cells.append(next_cell)
        mask_sequence.append(next_mask)
        current_cell = next_cell
        current_t = t

    else:
        status = "INVALID_MAX_STEPS"
        error = f"Exceeded MAX_STEPS={MAX_STEPS}"

    # Classify a physically connected exit by mask progression.
    if status == "BOUNDARY_EXIT":
        monotonic = all(
            mask_sequence[i] <= mask_sequence[i + 1]
            for i in range(len(mask_sequence) - 1)
        )
        exact_sequence = mask_sequence == EXPECTED_MASK_SEQUENCE
        if exact_sequence:
            final_status = "VALID"
        elif monotonic:
            final_status = "NON_UNIFORM"
        else:
            final_status = "NON_MONOTONIC"
    else:
        monotonic = False
        exact_sequence = False
        final_status = status or "UNKNOWN"

    return {
        "status": final_status,
        "trace_status": status or "",
        "error": error,
        "topology_exit_edge": exit_edge,
        "topology_exit_cell": exit_cell,
        "topology_exit_distance_km": exit_distance_km,
        "topology_exit_x_m": exit_x,
        "topology_exit_y_m": exit_y,
        "mask_sequence": "-".join(map(str, mask_sequence)),
        "monotonic": bool(monotonic),
        "exact_1_to_7_sequence": bool(exact_sequence),
        "visited_cells": ";".join(map(str, visited_cells)),
        "visited_edges": ";".join(map(str, visited_edges)),
        "n_cells_traversed": len(visited_cells),
        "n_edges_crossed": len(visited_edges),
        "global_nearest_edge": int(global_nearest_edge),
        "global_nearest_distance_km": float(global_nearest_distance_km),
        "topology_minus_global_km": (
            float(exit_distance_km - global_nearest_distance_km)
            if np.isfinite(exit_distance_km) and np.isfinite(global_nearest_distance_km)
            else np.nan
        ),
        "same_as_global_nearest_edge": (
            bool(exit_edge == global_nearest_edge) if exit_edge >= 0 else False
        ),
        "same_as_csv_network_outer_edge": (
            bool(exit_edge == network_outer_edge) if exit_edge >= 0 else False
        ),
    }


def find_global_nearest_physical_hit(origin, direction, edge_xy, physical_edges):
    hits = []
    for e in physical_edges:
        p1, p2 = edge_xy[e]
        if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
            continue
        hit = ray_segment_intersection(origin, direction, p1, p2)
        if hit is not None and hit[0] > INTERSECTION_TOL_M:
            hits.append((hit[0], int(e)))
    if not hits:
        return -1, np.nan
    hits.sort()
    return hits[0][1], hits[0][0] / 1000.0


def main():
    parser = argparse.ArgumentParser(
        description="Trace MPAS normal rays through cell-edge topology."
    )
    project_root = Path(__file__).resolve().parents[3]

    results_dir = (
        project_root
        / "boundary_analysis"
        / "results"
        / "topology"
        / "v1"
    )

    parser.add_argument(
        "--static",
        default=str(project_root / "iran.static.nc")
    )
    parser.add_argument(
        "--csv",
        default=str(
            project_root
            / "boundary_analysis"
            / "results"
            / "geometric"
            / "local_boundary_width.csv"
        )
    )
    parser.add_argument(
        "--output",
        default=str(results_dir / "boundary_ray_topology.csv")
    )
    parser.add_argument(
        "--report",
        default=str(results_dir / "boundary_ray_topology_report.txt")
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MPAS TOPOLOGY-AWARE NORMAL RAY TRACING")
    print("=" * 72)
    print(f"Static file : {args.static}")
    print(f"Input CSV   : {args.csv}")

    df = pd.read_csv(args.csv)
    required = {"inner_edge", "start_cell", "outer_edge"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CSV is missing required columns: {sorted(missing)}")

    with Dataset(args.static) as ds:
        lat_cell = as_float_array(first_existing(ds, ["latCell"]))
        lon_cell = as_float_array(first_existing(ds, ["lonCell"]))
        lat_vertex = as_float_array(first_existing(ds, ["latVertex"]))
        lon_vertex = as_float_array(first_existing(ds, ["lonVertex"]))
        cells_on_edge = np.asarray(first_existing(ds, ["cellsOnEdge"]), dtype=int)
        vertices_on_edge = np.asarray(first_existing(ds, ["verticesOnEdge"]), dtype=int)
        bdy_mask_cell = np.asarray(first_existing(ds, ["bdyMaskCell"]), dtype=int).ravel()

    # MPAS lat/lon variables are normally radians. Also tolerate files
    # whose coordinates are already stored in degrees.
    def degrees_if_radians(values, latitude=False):
        values = np.asarray(values, dtype=float)
        limit = (np.pi / 2 + 1e-3) if latitude else (2 * np.pi + 1e-3)
        if np.nanmax(np.abs(values)) <= limit:
            return np.rad2deg(values)
        return values

    lat_vertex = degrees_if_radians(lat_vertex, latitude=True)
    lon_vertex = degrees_if_radians(lon_vertex, latitude=False)
    lat_cell = degrees_if_radians(lat_cell, latitude=True)
    lon_cell = degrees_if_radians(lon_cell, latitude=False)

    n_cells = len(bdy_mask_cell)
    n_edges = cells_on_edge.shape[0]
    if cells_on_edge.shape[1] != 2 or vertices_on_edge.shape != (n_edges, 2):
        raise ValueError("Expected cellsOnEdge and verticesOnEdge dimensions (nEdges, 2)")
    if len(lat_vertex) != len(lon_vertex):
        raise ValueError("latVertex/lonVertex length mismatch")

    cell_edges = build_edge_cell_adjacency(cells_on_edge, n_cells)

    # Physical boundary edges: one valid adjacent cell, and that cell is Mask 7.
    physical_edges = []
    for e, pair in enumerate(cells_on_edge):
        valid = [int(v) - 1 for v in pair if int(v) > 0]
        if len(valid) == 1 and int(bdy_mask_cell[valid[0]]) == 7:
            physical_edges.append(e)

    print(f"Cells                 : {n_cells}")
    print(f"Edges                 : {n_edges}")
    print(f"Physical Mask-7 edges : {len(physical_edges)}")
    print(f"Input rays            : {len(df)}")

    results = []
    for row_number, row in df.iterrows():
        inner_edge = int(row["inner_edge"])
        start_cell = int(row["start_cell"])
        network_outer_edge = int(row["outer_edge"])

        base = {
            "inner_edge": inner_edge,
            "start_cell": start_cell,
            "start_mask": (
                int(bdy_mask_cell[start_cell])
                if 0 <= start_cell < n_cells else -1
            ),
            "csv_network_outer_edge": network_outer_edge,
        }

        try:
            lon0, lat0 = edge_midpoint_lonlat(
                inner_edge, vertices_on_edge, lat_vertex, lon_vertex
            )
            transformer = make_transformer(lon0, lat0)
            edge_xy = project_mesh(
                transformer, lon_vertex, lat_vertex, vertices_on_edge
            )
            origin = np.array([0.0, 0.0], dtype=float)

            # Project the two vertices of the inner edge.
            v1, v2 = map(int, vertices_on_edge[inner_edge])
            p1 = edge_xy[inner_edge, 0]
            p2 = edge_xy[inner_edge, 1]
            tangent = unit_vector(p2 - p1)
            normal = np.array([-tangent[1], tangent[0]], dtype=float)

            # Orient the normal toward the center of the CSV start cell.
            sx, sy = transformer.transform(
                float(lon_cell[start_cell]), float(lat_cell[start_cell])
            )
            start_xy = np.array([sx, sy], dtype=float)
            if float(np.dot(normal, start_xy - origin)) < 0.0:
                normal = -normal

            global_edge, global_dist = find_global_nearest_physical_hit(
                origin, normal, edge_xy, physical_edges
            )

            trace = trace_one_ray(
                inner_edge=inner_edge,
                start_cell=start_cell,
                network_outer_edge=network_outer_edge,
                cells_on_edge=cells_on_edge,
                bdy_mask_cell=bdy_mask_cell,
                cell_edges=cell_edges,
                edge_xy=edge_xy,
                origin=origin,
                direction=normal,
                global_nearest_edge=global_edge,
                global_nearest_distance_km=global_dist,
            )

            base.update({
                "origin_lon": lon0,
                "origin_lat": lat0,
                "normal_x": float(normal[0]),
                "normal_y": float(normal[1]),
                "global_nearest_edge": global_edge,
                "global_nearest_distance_km": global_dist,
            })
            base.update(trace)

            # Convert topological exit coordinate back to lon/lat.
            if np.isfinite(trace.get("topology_exit_x_m", np.nan)):
                inverse = Transformer.from_crs(
                    transformer.target_crs, "EPSG:4326", always_xy=True
                )
                ex_lon, ex_lat = inverse.transform(
                    trace["topology_exit_x_m"], trace["topology_exit_y_m"]
                )
                base["topology_exit_lon"] = float(ex_lon)
                base["topology_exit_lat"] = float(ex_lat)
            else:
                base["topology_exit_lon"] = np.nan
                base["topology_exit_lat"] = np.nan

        except Exception as exc:
            base.update({
                "status": "ERROR",
                "trace_status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "topology_exit_edge": -1,
                "topology_exit_cell": -1,
                "topology_exit_distance_km": np.nan,
                "mask_sequence": "",
                "monotonic": False,
                "exact_1_to_7_sequence": False,
                "visited_cells": "",
                "visited_edges": "",
                "n_cells_traversed": 0,
                "n_edges_crossed": 0,
                "global_nearest_edge": -1,
                "global_nearest_distance_km": np.nan,
                "topology_minus_global_km": np.nan,
                "same_as_global_nearest_edge": False,
                "same_as_csv_network_outer_edge": False,
            })

        results.append(base)

        if (len(results) % 25 == 0) or (len(results) == len(df)):
            print(f"Processed {len(results)}/{len(df)} rays")

    out = pd.DataFrame(results)
    out.to_csv(args.output, index=False)

    counts = out["status"].value_counts(dropna=False).to_dict()
    report_lines = [
        "=" * 72,
        "MPAS TOPOLOGY-AWARE NORMAL RAY TRACING REPORT",
        "=" * 72,
        f"Input rays: {len(out)}",
        f"Physical Mask-7 boundary edges: {len(physical_edges)}",
        "",
        "STATUS COUNTS",
        "-" * 72,
    ]
    for key in sorted(counts):
        report_lines.append(f"{key}: {counts[key]}")

    report_lines += [
        "",
        "VALIDATION SUMMARY",
        "-" * 72,
        f"VALID (exact 1-2-3-4-5-6-7): {(out['status'] == 'VALID').sum()}",
        f"NON_UNIFORM (connected, monotonic but not exact): {(out['status'] == 'NON_UNIFORM').sum()}",
        f"NON_MONOTONIC: {(out['status'] == 'NON_MONOTONIC').sum()}",
        f"BOUNDARY_EXIT: {(out['trace_status'] == 'BOUNDARY_EXIT').sum()}",
        f"NO_HIT: {(out['status'] == 'NO_HIT').sum()}",
        f"VERTEX_AMBIGUITY: {(out['status'] == 'VERTEX_AMBIGUITY').sum()}",
        f"Same edge as global nearest hit: {out['same_as_global_nearest_edge'].sum()}/{len(out)}",
        f"Same edge as CSV network exit: {out['same_as_csv_network_outer_edge'].sum()}/{len(out)}",
        "",
    ]

    valid_dist = out.loc[out["status"] == "VALID", "topology_exit_distance_km"].dropna()
    if len(valid_dist):
        report_lines += [
            "VALID WIDTH STATISTICS (km)",
            "-" * 72,
            f"n={len(valid_dist)}",
            f"mean={valid_dist.mean():.3f}",
            f"median={valid_dist.median():.3f}",
            f"min={valid_dist.min():.3f}",
            f"max={valid_dist.max():.3f}",
            "",
        ]
    else:
        report_lines += ["No VALID exact-sequence rays; no valid-width statistics.", ""]

    report_lines += ["SAMPLE RAYS", "-" * 72]
    for edge_id in [769, 777, 828]:
        sample = out[out["inner_edge"] == edge_id]
        if sample.empty:
            report_lines.append(f"Edge {edge_id}: not present")
            continue
        r = sample.iloc[0]
        report_lines.append(
            f"Edge {edge_id}: status={r.get('status')}, "
            f"masks={r.get('mask_sequence')}, "
            f"topology_exit_edge={r.get('topology_exit_edge')}, "
            f"topology_width_km={r.get('topology_exit_distance_km')}, "
            f"global_edge={r.get('global_nearest_edge')}, "
            f"global_width_km={r.get('global_nearest_distance_km')}, "
            f"CSV_edge={r.get('csv_network_outer_edge')}, "
            f"error={r.get('error', '')}"
        )
        report_lines.append(f"  cells: {r.get('visited_cells', '')}")

    report_lines += [
        "",
        "METHODOLOGICAL NOTES",
        "-" * 72,
        "1. The normal direction was not adjusted; it was oriented toward the Mask-1 start cell.",
        "2. Cell transitions use cellsOnEdge. Only edges incident to the current cell are considered.",
        "3. A boundary exit is accepted only when the current cell is Mask 7.",
        "4. Exact VALID status requires the complete sequence 1-2-3-4-5-6-7.",
        "5. Vertex ties are flagged as VERTEX_AMBIGUITY rather than resolved arbitrarily.",
        "6. Geometry is evaluated in a local AEQD projection centered on each inner-edge midpoint.",
        "   Projection distortion may become important for unusually long rays; inspect long paths separately.",
        "",
        f"CSV output: {args.output}",
    ]

    report = "\n".join(report_lines)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("\n" + report)
    print("\nFinished.")
    print(f"Detailed CSV : {args.output}")
    print(f"Text report  : {args.report}")


if __name__ == "__main__":
    main()
