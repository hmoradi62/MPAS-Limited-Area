from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

GRID_FILE = PROJECT_ROOT / "iran.static.nc"
WIDTH_FILE = (
    PROJECT_ROOT
    / "boundary_analysis"
    / "results"
    / "geometric"
    / "local_boundary_width.csv"
)

# --------------------------------------------------
# Load data
# --------------------------------------------------

ds = Dataset(GRID_FILE)

cells_on_edge = np.asarray(ds.variables["cellsOnEdge"][:])
cells_on_cell = np.asarray(ds.variables["cellsOnCell"][:])
cell_mask = np.asarray(ds.variables["bdyMaskCell"][:])

n_cells = len(cell_mask)
n_edges = len(cells_on_edge)

# --------------------------------------------------
# Identify physical boundary edges (zero-based)
# --------------------------------------------------

physical_edges = []

for edge_id, cells in enumerate(cells_on_edge):

    c1, c2 = map(int, cells)

    if (c1 == 0) == (c2 == 0):
        continue

    valid_cell = c2 if c1 == 0 else c1

    if 1 <= valid_cell <= n_cells:

        if cell_mask[valid_cell - 1] == 7:
            physical_edges.append(edge_id)

physical_edges = set(physical_edges)

# --------------------------------------------------
# Build cell -> physical boundary edges
# --------------------------------------------------

boundary_edges_by_cell = defaultdict(list)

for edge_id in physical_edges:

    cells = cells_on_edge[edge_id]

    for c in cells:

        c = int(c)

        if c > 0:
            cell_id = c - 1
            boundary_edges_by_cell[cell_id].append(edge_id)

# --------------------------------------------------
# Build cell adjacency
# --------------------------------------------------

neighbors = defaultdict(set)

for cell_id, row in enumerate(cells_on_cell):

    for value in row:

        value = int(value)

        if value > 0:

            neighbor_id = value - 1

            if neighbor_id != cell_id:
                neighbors[cell_id].add(neighbor_id)

# --------------------------------------------------
# Read local paths
# --------------------------------------------------

df = pd.read_csv(WIDTH_FILE)

print("=" * 75)
print("LOCAL PHYSICAL BOUNDARY TOPOLOGY")
print("=" * 75)

print("Physical boundary edges:", len(physical_edges))
print("Paths:", len(df))

# --------------------------------------------------
# Analyze first 10 paths
# --------------------------------------------------

for _, row in df.head(10).iterrows():

    inner_edge = int(row["inner_edge"])
    outer_edge = int(row["outer_edge"])
    outer_cell = int(row["outer_cell"])

    path = [
        int(x)
        for x in str(row["path_cell_ids"]).split(";")
    ]

    direct_edges = boundary_edges_by_cell.get(
        outer_cell, []
    )

    local_cells = {outer_cell}
    local_cells.update(neighbors[outer_cell])

    expanded_edges = set()

    for cell_id in local_cells:
        expanded_edges.update(
            boundary_edges_by_cell.get(cell_id, [])
        )

    print("\n" + "-" * 75)

    print("Inner edge:", inner_edge)
    print("Network outer edge:", outer_edge)
    print("Outer cell:", outer_cell)
    print("Path:", path)

    print("Boundary edges on outer cell:",
          sorted(direct_edges))

    print("Boundary edges on outer cell + neighbors:",
          sorted(expanded_edges))

    print("Network outer edge is physical boundary:",
          outer_edge in physical_edges)

    print("Network outer edge belongs to outer cell:",
          outer_edge in direct_edges)

    print("Network outer edge belongs to expanded neighborhood:",
          outer_edge in expanded_edges)

ds.close()
