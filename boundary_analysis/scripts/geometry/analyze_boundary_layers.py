from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


import numpy as np
from netCDF4 import Dataset

STATIC_FILE = PROJECT_ROOT / 'iran.static.nc'

cities = {
    "Kermanshah": (34.31, 47.06),
    "Tehran": (35.69, 51.39),
    "Tabriz": (38.08, 46.29),
    "Mashhad": (36.30, 59.60),
    "Zahedan": (29.50, 60.86),
    "Ahvaz": (31.32, 48.67),
    "Isfahan": (32.65, 51.67),
    "Shiraz": (29.59, 52.58),
}

EARTH_RADIUS_KM = 6371.0


def distance(lat1, lon1, lat2, lon2):

    # Convert independently to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(np.asarray(lat2))
    lon2 = np.radians(np.asarray(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2)**2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2)**2
    )

    a = np.clip(a, 0.0, 1.0)

    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ==================================================
# READ STATIC FILE
# ==================================================

with Dataset(STATIC_FILE) as nc:

    lat = np.degrees(nc.variables["latCell"][:])
    lon = np.degrees(nc.variables["lonCell"][:])

    mask = np.asarray(
        nc.variables["bdyMaskCell"][:]
    )

print("\nBOUNDARY LAYER ANALYSIS")
print("=" * 100)

print("Total cells:", len(lat))
print("Boundary layers:", np.unique(mask))

# ==================================================
# DISTANCE ANALYSIS
# ==================================================

for city, (clat, clon) in cities.items():

    distances = distance(
        clat,
        clon,
        lat,
        lon
    )

    print(f"\n{city}")
    print("-" * 75)

    for level in range(8):

        indices = np.where(mask == level)[0]

        if len(indices) == 0:
            print(f"Mask {level}: No cells")
            continue

        idx = indices[
            np.argmin(distances[indices])
        ]

        print(
            f"Mask {level}: "
            f"{distances[idx]:8.1f} km | "
            f"Cell {idx:4d} | "
            f"Lat {lat[idx]:7.3f} | "
            f"Lon {lon[idx]:7.3f}"
        )

print("\nANALYSIS COMPLETED")
