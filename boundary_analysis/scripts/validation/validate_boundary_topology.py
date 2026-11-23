from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CSV = (
    PROJECT_ROOT
    / "boundary_analysis"
    / "results"
    / "geometric"
    / "local_boundary_width.csv"
)

NC = PROJECT_ROOT / "iran.static.nc"

df = pd.read_csv(CSV)

with Dataset(NC) as nc:

    mask = np.asarray(nc["bdyMaskCell"][:])
    coe = np.asarray(nc["cellsOnEdge"][:])

    physical_edges = []
    internal_mask7_edges = []

    for i, cells in enumerate(coe):

        c1, c2 = map(int, cells)

        # Physical boundary:
        # One side has no cell, other side is Mask 7
        if c1 == 0 or c2 == 0:

            valid_cell = c2 if c1 == 0 else c1

            if valid_cell > 0 and mask[valid_cell - 1] == 7:
                physical_edges.append(i + 1)

        # Internal interface with Mask 7
        else:

            m1 = mask[c1 - 1]
            m2 = mask[c2 - 1]

            if (m1 == 7) != (m2 == 7):
                internal_mask7_edges.append(i + 1)

    csv_outer_edges = set(df["outer_edge"].astype(int))
    physical_set = set(physical_edges)

    print("=" * 65)
    print("CORRECTED PHYSICAL BOUNDARY VALIDATION")
    print("=" * 65)

    print("Physical boundary edges:", len(physical_edges))
    print("Expected:", 277)

    print("\nInternal Mask 7 interfaces:",
          len(internal_mask7_edges))

    print("\nCSV outer edges:", len(csv_outer_edges))

    print("CSV outer edges on physical boundary:",
          len(csv_outer_edges & physical_set))

    missing = csv_outer_edges - physical_set
    print("CSV edges NOT on physical boundary:", sorted(missing))

    print("\n===== FINAL CHECK =====")

    print(
        "Physical boundary count correct:",
        len(physical_edges) == 277
    )

    print(
        "All CSV outer edges valid:",
        csv_outer_edges.issubset(physical_set)
    )

    print(
        "All 207 outer edges unique:",
        len(csv_outer_edges) == 207
    )
