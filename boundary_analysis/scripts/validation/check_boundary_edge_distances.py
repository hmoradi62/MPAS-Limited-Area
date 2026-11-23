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
    / "normal_network"
    / "geometric_boundary_width.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "boundary_analysis"
    / "results"
    / "diagnostics"
    / "boundary_edge_distance_check.csv"
)

geod = Geod(ellps="WGS84")


def to_degrees(var):
    values = np.asarray(var[:], dtype=float)
    units = getattr(var, "units", "").lower()

    if "radian" in units:
        return np.degrees(values)

    if np.nanmax(np.abs(values)) <= 2 * np.pi + 0.1:
        return np.degrees(values)

    return values


def spherical_midpoint(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = np.radians(
        [lat1, lon1, lat2, lon2]
    )

    x = np.cos(lat1) * np.cos(lon1) + np.cos(lat2) * np.cos(lon2)
    y = np.cos(lat1) * np.sin(lon1) + np.cos(lat2) * np.sin(lon2)
    z = np.sin(lat1) + np.sin(lat2)

    lon = np.arctan2(y, x)
    hyp = np.sqrt(x*x + y*y)
    lat = np.arctan2(z, hyp)

    return np.degrees(lat), np.degrees(lon)


def edge_midpoint(edge_id, vertices_on_edge, lat_vertex, lon_vertex):
    idx = int(edge_id)

    vertices = vertices_on_edge[idx].astype(int) - 1

    if np.any(vertices < 0):
        raise ValueError(f"Invalid vertex index for edge {edge_id}")

    v1, v2 = vertices

    return spherical_midpoint(
        lat_vertex[v1], lon_vertex[v1],
        lat_vertex[v2], lon_vertex[v2]
    )


def distance_km(p1, p2):
    lat1, lon1 = p1
    lat2, lon2 = p2

    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    return dist / 1000.0


print("Loading data...")

df = pd.read_csv(CSV_FILE)

with Dataset(NC_FILE) as nc:
    lat_vertex = to_degrees(nc.variables["latVertex"])
    lon_vertex = to_degrees(nc.variables["lonVertex"])

    cells_on_edge = np.asarray(
        nc.variables["cellsOnEdge"][:], dtype=int
    )

    vertices_on_edge = np.asarray(
        nc.variables["verticesOnEdge"][:], dtype=int
    )

    mask_cell = np.asarray(
        nc.variables["bdyMaskCell"][:], dtype=int
    )

    # Physical boundary: one side is outside the mesh (0),
    # the other side belongs to Mask 7.
    physical_edges = []

    for i, cells in enumerate(cells_on_edge):
        c1, c2 = cells

        if (c1 == 0 and c2 > 0 and mask_cell[c2 - 1] == 7) or \
           (c2 == 0 and c1 > 0 and mask_cell[c1 - 1] == 7):
            physical_edges.append(i )

physical_edges = np.asarray(physical_edges, dtype=int)

print("Physical Mask 7 edges:", len(physical_edges))

# Calculate midpoint of every physical boundary edge.
physical_midpoints = {}

for edge_id in physical_edges:
    physical_midpoints[edge_id] = edge_midpoint(
        edge_id, vertices_on_edge, lat_vertex, lon_vertex
    )

records = []

print("Calculating independent geodesic distances...")

for _, row in df.iterrows():

    inner_id = int(row["inner_edge"])
    network_outer_id = int(row["network_outer_edge"])
    normal_outer_id = int(row["normal_outer_edge"])

    inner_mid = edge_midpoint(
        inner_id, vertices_on_edge, lat_vertex, lon_vertex
    )

    network_mid = edge_midpoint(
        network_outer_id, vertices_on_edge, lat_vertex, lon_vertex
    )

    normal_mid = edge_midpoint(
        normal_outer_id - 1 , vertices_on_edge, lat_vertex, lon_vertex
    )

    network_distance = distance_km(inner_mid, network_mid)
    normal_distance = distance_km(inner_mid, normal_mid)

    distances = []

    for edge_id, midpoint in physical_midpoints.items():
        d = distance_km(inner_mid, midpoint)
        distances.append((d, edge_id))

    distances.sort()

    nearest_distance, nearest_edge = distances[0]

    # Is the network destination among the physical boundary edges?
    network_is_physical = network_outer_id in physical_midpoints

    records.append({
        "inner_edge": inner_id,
        "network_outer_edge": network_outer_id,
        "normal_outer_edge": normal_outer_id,
        "network_outer_is_physical": network_is_physical,
        "network_midpoint_distance_km": network_distance,
        "normal_midpoint_distance_km": normal_distance,
        "nearest_physical_edge": nearest_edge,
        "nearest_physical_distance_km": nearest_distance,
        "nearest_physical_is_network_edge": (
            nearest_edge == network_outer_id
        ),
        "network_width_km": row["network_width_km"],
        "normal_width_km": row["normal_width_km"],
        "inner_lat": inner_mid[0],
        "inner_lon": inner_mid[1],
        "network_outer_lat": network_mid[0],
        "network_outer_lon": network_mid[1],
        "nearest_physical_lat": physical_midpoints[nearest_edge][0],
        "nearest_physical_lon": physical_midpoints[nearest_edge][1],
    })


out = pd.DataFrame(records)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_CSV, index=False)

print("\n===== DISTANCE SUMMARY =====")

for col in [
    "network_midpoint_distance_km",
    "normal_midpoint_distance_km",
    "nearest_physical_distance_km"
]:
    print(f"\n{col}")
    print(out[col].describe().round(3))

print("\n===== DESTINATION CHECK =====")
print(
    "Network destinations on physical boundary:",
    out["network_outer_is_physical"].sum(),
    "/", len(out)
)

print(
    "Nearest physical edge equals network destination:",
    out["nearest_physical_is_network_edge"].sum(),
    "/", len(out)
)

print("\n===== FIRST 20 ROWS =====")
print(
    out[[
        "inner_edge",
        "network_outer_edge",
        "nearest_physical_edge",
        "network_midpoint_distance_km",
        "nearest_physical_distance_km",
        "nearest_physical_is_network_edge"
    ]].head(20).round(3).to_string(index=False)
)

print("\nSaved: boundary_edge_distance_check.csv")
