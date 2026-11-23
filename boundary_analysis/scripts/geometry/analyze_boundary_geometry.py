from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


import heapq
import numpy as np
from netCDF4 import Dataset
from collections import Counter

STATIC_FILE = PROJECT_ROOT / 'iran.static.nc'
EARTH_RADIUS_KM = 6371.0


# ==========================================================
# 1. READ MPAS MESH
# ==========================================================

with Dataset(STATIC_FILE) as nc:

    lat = np.asarray(nc.variables["latCell"][:], dtype=float)
    lon = np.asarray(nc.variables["lonCell"][:], dtype=float)

    mask = np.asarray(
        nc.variables["bdyMaskCell"][:], dtype=int
    )

    cells_on_cell = np.asarray(
        nc.variables["cellsOnCell"][:], dtype=int
    )

n_cells = len(mask)

# Convert coordinates to radians
lat_rad = np.radians(lat)
lon_rad = np.radians(lon)


# ==========================================================
# 2. BUILD NEIGHBOR GRAPH
# ==========================================================

neighbors = [[] for _ in range(n_cells)]

for i in range(n_cells):

    for value in cells_on_cell[i]:

        if value <= 0:
            continue

        j = value - 1

        if j == i:
            continue

        neighbors[i].append(j)


# ==========================================================
# 3. GEODESIC DISTANCE BETWEEN CELL CENTERS
# ==========================================================

def haversine(i, j):

    dlat = lat_rad[j] - lat_rad[i]
    dlon = lon_rad[j] - lon_rad[i]

    a = (
        np.sin(dlat / 2)**2
        + np.cos(lat_rad[i])
        * np.cos(lat_rad[j])
        * np.sin(dlon / 2)**2
    )

    a = np.clip(a, 0.0, 1.0)

    return (
        2 * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(a))
    )


# ==========================================================
# 4. CONNECTED COMPONENTS FOR EACH MASK CLASS
# ==========================================================

def connected_components(level):

    selected = set(np.where(mask == level)[0])
    visited = set()
    components = []

    for start in selected:

        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        component = []

        while stack:

            i = stack.pop()
            component.append(i)

            for j in neighbors[i]:

                if j in selected and j not in visited:

                    visited.add(j)
                    stack.append(j)

        components.append(component)

    return components


print("\n" + "=" * 75)
print("MPAS BOUNDARY CONNECTED COMPONENT ANALYSIS")
print("=" * 75)

component_results = {}

for level in range(8):

    components = connected_components(level)
    component_results[level] = components

    sizes = sorted(
        [len(c) for c in components],
        reverse=True
    )

    print(f"\nMask {level}")
    print("Number of components:", len(components))
    print("Component sizes:", sizes)

    if len(sizes) > 1:

        print(
            "WARNING: This mask class contains "
            "multiple disconnected components."
        )


# ==========================================================
# 5. VERIFY ADJACENCY BETWEEN CONSECUTIVE CLASSES
# ==========================================================

print("\n" + "=" * 75)
print("CONSECUTIVE MASK CONNECTIVITY")
print("=" * 75)

for level in range(7):

    count = 0

    for i in range(n_cells):

        if mask[i] != level:
            continue

        for j in neighbors[i]:

            if mask[j] == level + 1:
                count += 1

    print(
        f"Mask {level} -> Mask {level+1}: "
        f"{count} directed connections"
    )


# ==========================================================
# 6. MULTI-SOURCE DIJKSTRA
# Distance from internal cells to boundary layers
# Paths are restricted to Mask >= 1 after leaving Mask 0.
# ==========================================================

print("\n" + "=" * 75)
print("GEOMETRIC BOUNDARY WIDTH ANALYSIS")
print("=" * 75)

dist = np.full(n_cells, np.inf)
previous = np.full(n_cells, -1, dtype=int)

queue = []

sources = np.where(mask == 0)[0]

for i in sources:

    dist[i] = 0.0
    heapq.heappush(queue, (0.0, int(i)))


while queue:

    current_dist, i = heapq.heappop(queue)

    if current_dist > dist[i]:
        continue

    for j in neighbors[i]:

        # Once the path leaves the interior,
        # it may only continue through boundary cells.
        if mask[j] == 0:
            continue

        weight = haversine(i, j)
        new_dist = current_dist + weight

        if new_dist < dist[j]:

            dist[j] = new_dist
            previous[j] = i

            heapq.heappush(
                queue,
                (new_dist, j)
            )


# ==========================================================
# 7. DISTANCE DISTRIBUTION BY MASK CLASS
# ==========================================================

print("\nDISTANCE FROM INTERIOR MASK=0")

for level in range(1, 8):

    indices = np.where(mask == level)[0]
    values = dist[indices]

    valid = values[np.isfinite(values)]

    if len(valid) == 0:

        print(f"Mask {level}: No reachable cells")
        continue

    print(
        f"Mask {level}: "
        f"min={np.min(valid):.2f} km | "
        f"median={np.median(valid):.2f} km | "
        f"mean={np.mean(valid):.2f} km | "
        f"max={np.max(valid):.2f} km"
    )


# ==========================================================
# 8. ESTIMATE WIDTH AT OUTER BOUNDARY
# ==========================================================

outer_indices = np.where(mask == 7)[0]
outer_distances = dist[outer_indices]

valid_outer = outer_distances[
    np.isfinite(outer_distances)
]

print("\n" + "=" * 75)
print("BOUNDARY WIDTH SUMMARY")
print("=" * 75)

print("Reachable Mask 7 cells:", len(valid_outer))
print("Total Mask 7 cells:", len(outer_indices))

if len(valid_outer):

    print(
        "Minimum center-path width:",
        f"{valid_outer.min():.2f} km"
    )

    print(
        "Median center-path width:",
        f"{np.median(valid_outer):.2f} km"
    )

    print(
        "Mean center-path width:",
        f"{valid_outer.mean():.2f} km"
    )

    print(
        "Maximum center-path width:",
        f"{valid_outer.max():.2f} km"
    )


# ==========================================================
# 9. TRACE LONGEST AND SHORTEST PATHS
# ==========================================================

def trace_path(target):

    path = []
    current = int(target)

    while current >= 0:

        path.append(current)
        current = previous[current]

    return path[::-1]


if len(valid_outer):

    print("\nPATH EXAMPLES")

    for label, target in (
        ("MINIMUM", outer_indices[np.argmin(outer_distances)]),
        ("MEDIAN", outer_indices[
            np.argmin(np.abs(outer_distances - np.median(valid_outer)))
        ]),
        ("MAXIMUM", outer_indices[np.argmax(outer_distances)])
    ):

        path = trace_path(target)

        print(f"\n{label} PATH")
        print("Cell count:", len(path))
        print("Path distance:", f"{dist[target]:.2f} km")
        print(
            "Mask sequence:",
            [int(mask[i]) for i in path]
        )


print("\nANALYSIS COMPLETED")
