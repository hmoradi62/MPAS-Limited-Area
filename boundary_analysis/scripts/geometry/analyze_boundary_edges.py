from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree

STATIC_FILE = PROJECT_ROOT / 'iran.static.nc'
R = 6371.0  # Earth radius in km

with Dataset(STATIC_FILE) as nc:
    lat_edge = np.asarray(nc["latEdge"][:], dtype=float)
    lon_edge = np.asarray(nc["lonEdge"][:], dtype=float)
    cells_edge = np.asarray(nc["cellsOnEdge"][:], dtype=int)
    dv_edge = np.asarray(nc["dvEdge"][:], dtype=float)
    mask = np.asarray(nc["bdyMaskCell"][:], dtype=int)

# MPAS indices are one-based; zero means no cell.
# Physical boundary edges have exactly one adjacent cell.
physical = np.any(cells_edge == 0, axis=1)

# Internal interfaces between mask classes 0 and 1.
c0 = cells_edge[:, 0] - 1
c1 = cells_edge[:, 1] - 1

valid = (c0 >= 0) & (c1 >= 0)

inner = np.zeros(len(cells_edge), dtype=bool)
inner[valid] = (
    ((mask[c0[valid]] == 0) & (mask[c1[valid]] == 1)) |
    ((mask[c0[valid]] == 1) & (mask[c1[valid]] == 0))
)

# Physical boundary edges adjacent to each mask class.
boundary_mask = np.full(len(cells_edge), -1, dtype=int)
for e in np.where(physical)[0]:
    cell = c0[e] if c0[e] >= 0 else c1[e]
    if cell >= 0:
        boundary_mask[e] = mask[cell]

outer7 = physical & (boundary_mask == 7)

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = np.radians(
        [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2)**2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    )
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

print("\n" + "=" * 70)
print("PHYSICAL DOMAIN BOUNDARY")
print("=" * 70)

physical_ids = np.where(physical)[0]
outer7_ids = np.where(outer7)[0]

print("Total physical boundary edges:", len(physical_ids))
print("Boundary edges adjacent to Mask 7:", len(outer7_ids))
print("Boundary edges adjacent to other masks:",
      len(physical_ids) - len(outer7_ids))

if len(physical_ids):
    coverage = 100 * len(outer7_ids) / len(physical_ids)
    print(f"Mask 7 boundary coverage: {coverage:.2f}%")

    unique, counts = np.unique(
        boundary_mask[physical_ids], return_counts=True
    )
    print("\nBoundary edge distribution:")
    for level, count in zip(unique, counts):
        print(f"Mask {level}: {count} edges")

print("\n" + "=" * 70)
print("MASK 0 - MASK 1 INTERFACE")
print("=" * 70)

inner_ids = np.where(inner)[0]

print("Number of interface edges:", len(inner_ids))
print("Total interface length (km):",
      np.sum(dv_edge[inner_ids]) / 1000)

print("\n" + "=" * 70)
print("MASK 7 - PHYSICAL BOUNDARY")
print("=" * 70)

print("Number of Mask 7 outer edges:", len(outer7_ids))
print("Total outer boundary length (km):",
      np.sum(dv_edge[outer7_ids]) / 1000)

# Convert edge coordinates to 3-D unit sphere coordinates.
def xyz(lat, lon):
    lat = np.radians(lat)
    lon = np.radians(lon)
    return np.column_stack((
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ))

print("\n" + "=" * 70)
print("NEAREST INTERFACE-TO-OUTER-BOUNDARY DISTANCE")
print("=" * 70)

if len(inner_ids) and len(outer7_ids):
    inner_xyz = xyz(lat_edge[inner_ids], lon_edge[inner_ids])
    outer_xyz = xyz(lat_edge[outer7_ids], lon_edge[outer7_ids])

    tree = cKDTree(outer_xyz)
    chord, nearest = tree.query(inner_xyz, k=1)

    angular = 2 * np.arcsin(np.clip(chord / 2, 0, 1))
    distances = R * angular

    print("Minimum distance (km):", f"{distances.min():.2f}")
    print("Median distance (km):", f"{np.median(distances):.2f}")
    print("Mean distance (km):", f"{distances.mean():.2f}")
    print("Maximum distance (km):", f"{distances.max():.2f}")

    print("\nDistance percentiles:")
    for p in [5, 25, 50, 75, 95]:
        print(f"P{p}: {np.percentile(distances, p):.2f} km")

else:
    print("Cannot calculate: one of the edge sets is empty.")

print("\nANALYSIS COMPLETED")
