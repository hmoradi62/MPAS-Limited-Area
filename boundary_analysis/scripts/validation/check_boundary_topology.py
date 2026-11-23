
from pathlib import Path
from collections import defaultdict

import numpy as np
from netCDF4 import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_FILE = PROJECT_ROOT / "iran.static.nc"

with Dataset(STATIC_FILE) as nc:

    mask = np.asarray(nc.variables["bdyMaskCell"][:])
    cells_on_cell = np.asarray(nc.variables["cellsOnCell"][:])

n_cells = len(mask)

# MPAS indices are one-based; zero means no neighbor.
neighbors = []

for i in range(n_cells):
    row = cells_on_cell[i]
    valid = row[row > 0] - 1
    neighbors.append(valid)

print("\nMPAS BOUNDARY TOPOLOGY")
print("=" * 75)

# --------------------------------------------------
# 1. Count cells in each mask class
# --------------------------------------------------

print("\nCELL COUNTS")

for level in range(8):
    count = np.count_nonzero(mask == level)
    print(f"Mask {level}: {count:4d}")

# --------------------------------------------------
# 2. Count adjacency between mask classes
# --------------------------------------------------

pairs = defaultdict(int)

for i in range(n_cells):

    for j in neighbors[i]:

        if j <= i:
            continue

        a = int(mask[i])
        b = int(mask[j])

        pair = tuple(sorted((a, b)))
        pairs[pair] += 1

print("\nADJACENT MASK CLASS PAIRS")
print("-" * 75)

for pair in sorted(pairs):
    print(
        f"Mask {pair[0]} <-> Mask {pair[1]}: "
        f"{pairs[pair]}"
    )

# --------------------------------------------------
# 3. Check transitions between consecutive layers
# --------------------------------------------------

print("\nCONSECUTIVE LAYER CONNECTIONS")
print("-" * 75)

for level in range(7):

    count = pairs.get((level, level + 1), 0)

    print(
        f"Mask {level} <-> Mask {level+1}: "
        f"{count} shared cell edges"
    )

# --------------------------------------------------
# 4. Number of neighbors by mask class
# --------------------------------------------------

print("\nNEIGHBOR MASK DISTRIBUTION")

for level in range(8):

    cells = np.where(mask == level)[0]

    if len(cells) == 0:
        continue

    neighbor_levels = defaultdict(int)

    for i in cells:
        for j in neighbors[i]:
            neighbor_levels[int(mask[j])] += 1

    print(f"\nMask {level}:")

    for other in sorted(neighbor_levels):
        print(
            f"  Neighbor Mask {other}: "
            f"{neighbor_levels[other]}"
        )

print("\nTOPOLOGY ANALYSIS COMPLETED")
