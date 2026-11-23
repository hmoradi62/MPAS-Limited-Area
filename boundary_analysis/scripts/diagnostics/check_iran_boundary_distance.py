
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_FILE = PROJECT_ROOT / "iran.static.nc"

# Coordinates: latitude, longitude
cities = {
    "Kermanshah": (34.31, 47.06),
    "Tehran": (35.69, 51.39),
    "Tabriz": (38.08, 46.29),
    "Mashhad": (36.30, 59.60),
    "Zahedan": (29.50, 60.86),
    "Bandar Abbas": (27.18, 56.27),
    "Ahvaz": (31.32, 48.67),
    "Rasht": (37.28, 49.58),
    "Urmia": (37.55, 45.08),
    "Bushehr": (28.97, 50.84),
}

EARTH_RADIUS_KM = 6371.0


def haversine(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2)**2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2)**2
    )

    return 2 * EARTH_RADIUS_KM * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


with Dataset(STATIC_FILE) as nc:

    lat = np.degrees(nc.variables["latCell"][:])
    lon = np.degrees(nc.variables["lonCell"][:])

    mask = np.asarray(nc.variables["bdyMaskCell"][:])

boundary = mask > 0
interior = mask == 0

print("\nMPAS DOMAIN BOUNDARY DISTANCE ANALYSIS")
print("=" * 90)

print(
    f"{'City':18s} {'Nearest boundary':>18s} "
    f"{'Mask':>8s} {'Nearest interior':>18s}"
)

print("-" * 90)

for city, (city_lat, city_lon) in cities.items():

    distances = haversine(
        city_lat,
        city_lon,
        lat,
        lon
    )

    boundary_indices = np.where(boundary)[0]
    interior_indices = np.where(interior)[0]

    ib = boundary_indices[
        np.argmin(distances[boundary_indices])
    ]

    ii = interior_indices[
        np.argmin(distances[interior_indices])
    ]

    print(
        f"{city:18s} "
        f"{distances[ib]:14.1f} km "
        f"{mask[ib]:8d} "
        f"{distances[ii]:14.1f} km"
    )

    print(
        f"    Boundary cell: "
        f"lat={lat[ib]:.3f}, lon={lon[ib]:.3f}"
    )
