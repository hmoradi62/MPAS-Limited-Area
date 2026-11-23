import re
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import Geod

PROJECT_ROOT = Path(__file__).resolve().parents[3]

NC_FILE = PROJECT_ROOT / "iran.static.nc"

CSV_FILE = (
    PROJECT_ROOT
    / "boundary_analysis"
    / "results"
    / "geometric"
    / "local_boundary_width.csv"
)

geod = Geod(ellps="WGS84")


def read_degrees(var):
    a = np.asarray(var[:], dtype=float)
    units = getattr(var, "units", "").lower()

    if "radian" in units:
        return np.degrees(a)

    if np.nanmax(np.abs(a)) <= 2 * np.pi + 0.1:
        return np.degrees(a)

    return a


def dist(lat1, lon1, lat2, lon2):
    _, _, d = geod.inv(lon1, lat1, lon2, lat2)
    return d / 1000


def parse_cells(value):
    # Extract integer cell IDs from path_cells.
    return [int(x) for x in re.findall(r"\d+", str(value))]


df = pd.read_csv(CSV_FILE)

print("CSV columns:")
print(df.columns.tolist())

with Dataset(NC_FILE) as nc:

    lat_cell = read_degrees(nc.variables["latCell"])
    lon_cell = read_degrees(nc.variables["lonCell"])

    lat_vertex = read_degrees(nc.variables["latVertex"])
    lon_vertex = read_degrees(nc.variables["lonVertex"])

    cells_on_edge = np.asarray(
        nc.variables["cellsOnEdge"][:], dtype=int
    )

    vertices_on_edge = np.asarray(
        nc.variables["verticesOnEdge"][:], dtype=int
    )

    mask = np.asarray(nc.variables["bdyMaskCell"][:], dtype=int)

    print("\n===== COORDINATE RANGES =====")
    print("latCell:", np.nanmin(lat_cell), np.nanmax(lat_cell))
    print("lonCell:", np.nanmin(lon_cell), np.nanmax(lon_cell))
    print("latVertex:", np.nanmin(lat_vertex), np.nanmax(lat_vertex))
    print("lonVertex:", np.nanmin(lon_vertex), np.nanmax(lon_vertex))

    def edge_info(edge_id):
        i = int(edge_id) - 1

        c = cells_on_edge[i]
        v = vertices_on_edge[i]

        c_valid = [int(x) for x in c if x > 0]
        v_valid = [int(x) for x in v if x > 0]

        print(f"\nEdge {edge_id}")
        print("  cellsOnEdge:", c)
        print("  cell masks:", [
            int(mask[x - 1]) for x in c_valid
        ])

        print("  verticesOnEdge:", v)

        if len(v_valid) == 2:
            v1, v2 = [x - 1 for x in v_valid]

            length = dist(
                lat_vertex[v1], lon_vertex[v1],
                lat_vertex[v2], lon_vertex[v2]
            )

            print("  vertex 1:", lat_vertex[v1], lon_vertex[v1])
            print("  vertex 2:", lat_vertex[v2], lon_vertex[v2])
            print("  edge length km:", length)

        for cell_id in c_valid:
            j = cell_id - 1
            print(
                f"  cell {cell_id}: "
                f"lat={lat_cell[j]:.5f}, "
                f"lon={lon_cell[j]:.5f}, "
                f"mask={mask[j]}"
            )

    print("\n===== FIRST 5 PATHS =====")

    for _, row in df.head(5).iterrows():

        inner = int(row["inner_edge"])
        outer = int(row["outer_edge"])
        start = int(row["start_cell"])
        outer_cell = int(row["outer_cell"])

        print("\n" + "=" * 65)
        print(
            f"INNER={inner}, START={start}, "
            f"OUTER={outer}, OUTER_CELL={outer_cell}"
        )

        edge_info(inner)
        edge_info(outer)

        print("\nPath cells:")

        cells = parse_cells(row["path_cells"])

        print("  Number of cells:", len(cells))
        print("  First cells:", cells[:10])
        print("  Last cells:", cells[-10:])

        if not cells:
            print("  ERROR: Could not parse path_cells")
            continue

        print(
            "  First cell equals start_cell:",
            cells[0] == start
        )

        print(
            "  Last cell equals outer_cell:",
            cells[-1] == outer_cell
        )

        invalid = [
            c for c in cells if c < 1 or c > len(lat_cell)
        ]

        print("  Invalid cell IDs:", invalid[:10])

        if invalid:
            continue

        segment_distances = []

        for a, b in zip(cells[:-1], cells[1:]):
            d = dist(
                lat_cell[a - 1], lon_cell[a - 1],
                lat_cell[b - 1], lon_cell[b - 1]
            )
            segment_distances.append(d)

        print("  First cell coordinates:",
              lat_cell[cells[0]-1], lon_cell[cells[0]-1])

        print("  Last cell coordinates:",
              lat_cell[cells[-1]-1], lon_cell[cells[-1]-1])

        if segment_distances:
            print(
                "  Sum consecutive cell distances km:",
                sum(segment_distances)
            )
            print(
                "  Max consecutive cell distance km:",
                max(segment_distances)
            )
            print(
                "  Mean consecutive cell distance km:",
                np.mean(segment_distances)
            )

        print(
            "  Start-to-end cell distance km:",
            dist(
                lat_cell[cells[0]-1], lon_cell[cells[0]-1],
                lat_cell[cells[-1]-1], lon_cell[cells[-1]-1]
            )
        )

        print("  Mask sequence:",
              [int(mask[c - 1]) for c in cells])


print("\n===== DIAGNOSTIC COMPLETED =====")
