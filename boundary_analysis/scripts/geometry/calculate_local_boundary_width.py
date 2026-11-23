
import heapq
import csv
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

STATIC_FILE = PROJECT_ROOT / "iran.static.nc"
OUTPUT_CSV = (
    PROJECT_ROOT
    / "boundary_analysis"
    / "results"
    / "geometric"
    / "local_boundary_width.csv"
)

EARTH_RADIUS_KM = 6371.0

# ============================================================
# READ MPAS MESH
# ============================================================
with Dataset(STATIC_FILE) as nc:

    # MPAS coordinates are stored in radians.
    # Convert all coordinates to degrees before distance calculations.

    lat_cell = np.degrees(
        np.asarray(nc["latCell"][:], dtype=float)
    )

    lon_cell = np.degrees(
        np.asarray(nc["lonCell"][:], dtype=float)
    )

    lat_edge = np.degrees(
        np.asarray(nc["latEdge"][:], dtype=float)
    )

    lon_edge = np.degrees(
        np.asarray(nc["lonEdge"][:], dtype=float)
    )

    mask = np.asarray(
        nc["bdyMaskCell"][:], dtype=int
    )

    cells_on_edge = np.asarray(
        nc["cellsOnEdge"][:], dtype=int
    )

    cells_on_cell = np.asarray(
        nc["cellsOnCell"][:], dtype=int
    )


# ==========================================================
# DISTANCE FUNCTION
# ==========================================================
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    return (
        2.0 * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(a))
    )


# ============================================================
# BUILD CELL NEIGHBOR GRAPH
# ============================================================
n_cells = len(lat_cell)
neighbors = [[] for _ in range(n_cells)]

for i in range(n_cells):
    for value in cells_on_cell[i]:
        if value <= 0:
            continue

        j = int(value) - 1

        if j != i and j not in neighbors[i]:
            neighbors[i].append(j)


# ============================================================
# IDENTIFY INNER INTERFACE: MASK 0 / MASK 1
# ============================================================

inner_edges = []

for e, (c1_raw, c2_raw) in enumerate(cells_on_edge):

    if c1_raw <= 0 or c2_raw <= 0:
        continue

    c1 = int(c1_raw) - 1
    c2 = int(c2_raw) - 1

    m1 = mask[c1]
    m2 = mask[c2]

    if (m1 == 0 and m2 == 1):
        inner_edges.append((e, c2))

    elif (m1 == 1 and m2 == 0):
        inner_edges.append((e, c1))


# ============================================================
# IDENTIFY PHYSICAL BOUNDARY EDGES BY CELL
# ============================================================

boundary_edges_by_cell = [[] for _ in range(n_cells)]

for e, (c1_raw, c2_raw) in enumerate(cells_on_edge):

    # Physical boundary: one adjacent cell is zero
    if (c1_raw == 0) == (c2_raw == 0):
        continue

    cell_raw = c2_raw if c1_raw == 0 else c1_raw

    if cell_raw <= 0:
        continue

    cell = int(cell_raw) - 1

    if mask[cell] == 7:
        boundary_edges_by_cell[cell].append(e)


# ============================================================
# OUTWARD CONSTRAINED DIJKSTRA
# ============================================================

def find_outward_path(inner_edge, start_cell):

    # Initial distance: inner interface edge -> Mask 1 cell center
    initial_distance = haversine(
        lat_edge[inner_edge],
        lon_edge[inner_edge],
        lat_cell[start_cell],
        lon_cell[start_cell]
    )

    distances = np.full(n_cells, np.inf)
    previous = np.full(n_cells, -1, dtype=int)

    distances[start_cell] = initial_distance

    queue = [(initial_distance, start_cell)]

    best_distance = np.inf
    best_outer_edge = -1
    best_cell = -1

    while queue:

        current_distance, cell = heapq.heappop(queue)

        if current_distance > distances[cell]:
            continue

        if current_distance >= best_distance:
            continue

        # Check whether this Mask 7 cell touches physical boundary
        if mask[cell] == 7:

            for outer_edge in boundary_edges_by_cell[cell]:

                terminal_distance = haversine(
                    lat_cell[cell],
                    lon_cell[cell],
                    lat_edge[outer_edge],
                    lon_edge[outer_edge]
                )

                total_distance = (
                    current_distance + terminal_distance
                )

                if total_distance < best_distance:
                    best_distance = total_distance
                    best_outer_edge = outer_edge
                    best_cell = cell

        # Continue only outward or within the same mask layer
        for nb in neighbors[cell]:

            if mask[nb] < mask[cell]:
                continue

            if mask[nb] > 7:
                continue

            step_distance = haversine(
                lat_cell[cell],
                lon_cell[cell],
                lat_cell[nb],
                lon_cell[nb]
            )

            new_distance = current_distance + step_distance

            if new_distance < distances[nb]:
                distances[nb] = new_distance
                previous[nb] = cell

                heapq.heappush(
                    queue,
                    (new_distance, nb)
                )

    if best_outer_edge < 0:
        return None

    # Reconstruct cell path
    path = []
    current = best_cell

    while current >= 0:
        path.append(current)
        current = previous[current]

    path.reverse()

    return {
        "width_km": best_distance,
        "outer_edge": best_outer_edge,
        "outer_cell": best_cell,
        "path": path
    }


# ============================================================
# PROCESS ALL INNER INTERFACE EDGES
# ============================================================

results = []

print("=" * 75)
print("LOCAL MPAS BOUNDARY WIDTH ANALYSIS")
print("=" * 75)

print("Number of inner interface edges:", len(inner_edges))

for index, (inner_edge, start_cell) in enumerate(inner_edges, start=1):

    result = find_outward_path(inner_edge, start_cell)

    if result is None:
        print(
            f"WARNING: No outward path for inner edge {inner_edge}"
        )
        continue

    path = result["path"]

    mask_sequence = [int(mask[c]) for c in path]

    # Check expected layer sequence
    expected = all(
        mask_sequence[i] <= mask_sequence[i + 1]
        for i in range(len(mask_sequence) - 1)
    )

    results.append({
        "inner_edge": inner_edge,
        "start_cell": start_cell,
        "outer_edge": result["outer_edge"],
        "outer_cell": result["outer_cell"],
        "width_km": result["width_km"],
        "path_cells": len(path),
        "path_cell_ids": ";".join(map(str, path)),
        "mask_sequence": "-".join(map(str, mask_sequence)),
        "monotonic": expected
    })

    if index % 25 == 0:
        print(f"Processed {index}/{len(inner_edges)}")


# ============================================================
# SUMMARY
# ============================================================

widths = np.array(
    [item["width_km"] for item in results],
    dtype=float
)

print("\n" + "=" * 75)
print("LOCAL WIDTH STATISTICS")
print("=" * 75)

print("Successful paths:", len(results))
print("Failed paths:", len(inner_edges) - len(results))

if len(widths):

    print(f"Minimum : {widths.min():.2f} km")
    print(f"P05     : {np.percentile(widths, 5):.2f} km")
    print(f"P25     : {np.percentile(widths, 25):.2f} km")
    print(f"Median  : {np.median(widths):.2f} km")
    print(f"Mean    : {widths.mean():.2f} km")
    print(f"P75     : {np.percentile(widths, 75):.2f} km")
    print(f"P95     : {np.percentile(widths, 95):.2f} km")
    print(f"Maximum : {widths.max():.2f} km")

    print("\nMask sequence monotonic:",
          all(item["monotonic"] for item in results))


# ============================================================
# SAVE CSV
# ============================================================
# All edge and cell IDs in CSV are ZERO-BASED Python indices.
# NetCDF connectivity arrays (cellsOnEdge, etc.) remain ONE-BASED.
fieldnames = [
    "inner_edge",
    "start_cell",
    "outer_edge",
    "outer_cell",
    "width_km",
    "path_cells",
    "mask_sequence",
    "path_cell_ids",
    "monotonic"
]

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

print("\nSaved:", OUTPUT_CSV)
print("ANALYSIS COMPLETED")
