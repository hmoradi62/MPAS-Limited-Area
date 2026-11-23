
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from netCDF4 import Dataset
from matplotlib.collections import PolyCollection

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ==========================================================
# PATHS
# ==========================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATIC_FILE = PROJECT_ROOT / "iran.static.nc"

OUTPUT_DIR = PROJECT_ROOT / "domain_analysis" / "results"

OUTPUT_FILE = OUTPUT_DIR / "mpas_domain_coverage.png"


# ==========================================================
# READ MPAS STATIC FILE
# ==========================================================

with Dataset(STATIC_FILE) as nc:

    lat_cell = np.degrees(nc.variables["latCell"][:])
    lon_cell = np.degrees(nc.variables["lonCell"][:])

    lat_vertex = np.degrees(nc.variables["latVertex"][:])
    lon_vertex = np.degrees(nc.variables["lonVertex"][:])

    vertices_on_cell = np.asarray(
        nc.variables["verticesOnCell"][:]
    )

    n_edges_on_cell = np.asarray(
        nc.variables["nEdgesOnCell"][:]
    )

    boundary_mask = np.asarray(
        nc.variables["bdyMaskCell"][:]
    )

    n_cells = len(lat_cell)


# ==========================================================
# BUILD CELL POLYGONS
# ==========================================================

polygons = []

for i in range(n_cells):

    n = int(n_edges_on_cell[i])

    # MPAS indices are one-based.
    vertex_ids = vertices_on_cell[i, :n] - 1

    polygon = np.column_stack((
        lon_vertex[vertex_ids],
        lat_vertex[vertex_ids]
    ))

    polygons.append(polygon)


# ==========================================================
# DOMAIN INFORMATION
# ==========================================================

print("\n========== MPAS DOMAIN ==========")

print("Number of cells:", n_cells)

print(
    "Latitude range:",
    f"{lat_cell.min():.3f} to {lat_cell.max():.3f}"
)

print(
    "Longitude range:",
    f"{lon_cell.min():.3f} to {lon_cell.max():.3f}"
)

print("\nBoundary mask distribution:")

values, counts = np.unique(
    boundary_mask,
    return_counts=True
)

for value, count in zip(values, counts):

    print(
        f"Mask {value}: {count} cells "
        f"({100 * count / n_cells:.2f}%)"
    )


# ==========================================================
# FIGURE
# ==========================================================

fig = plt.figure(
    figsize=(17, 11),
    dpi=180
)

projection = ccrs.PlateCarree()


# ==========================================================
# PANEL 1: REGIONAL DOMAIN
# ==========================================================

ax1 = fig.add_subplot(
    2, 1, 1,
    projection=projection
)

ax1.set_extent(
    [20, 80, 8, 58],
    crs=projection
)

ax1.add_feature(
    cfeature.LAND,
    facecolor="#f3f0e8",
    zorder=0
)

ax1.add_feature(
    cfeature.OCEAN,
    facecolor="#e5f1f7",
    zorder=0
)

ax1.add_feature(
    cfeature.COASTLINE,
    linewidth=0.6
)

ax1.add_feature(
    cfeature.BORDERS,
    linewidth=0.5,
    linestyle=":"
)

ax1.add_feature(
    cfeature.LAKES,
    facecolor="#e5f1f7",
    edgecolor="gray",
    linewidth=0.3
)

# All MPAS cells
mesh = PolyCollection(
    polygons,
    facecolors="none",
    edgecolors="gray",
    linewidths=0.25,
    transform=projection,
    zorder=2
)

ax1.add_collection(mesh)

# Boundary cells
boundary_polygons = [
    polygons[i]
    for i in range(n_cells)
    if boundary_mask[i] > 0
]

boundary_mesh = PolyCollection(
    boundary_polygons,
    facecolors="none",
    edgecolors="red",
    linewidths=0.55,
    transform=projection,
    zorder=3
)

ax1.add_collection(boundary_mesh)

# Cell centers
ax1.scatter(
    lon_cell,
    lat_cell,
    s=1.5,
    color="black",
    transform=projection,
    zorder=4
)

ax1.set_title(
    "MPAS Limited-Area Domain - Western Asia",
    fontsize=14,
    weight="bold"
)

gl = ax1.gridlines(
    draw_labels=True,
    linewidth=0.4,
    color="gray",
    alpha=0.5,
    linestyle="--"
)

gl.top_labels = False
gl.right_labels = False


# ==========================================================
# PANEL 2: IRAN AND NEIGHBORING COUNTRIES
# ==========================================================

ax2 = fig.add_subplot(
    2, 1, 2,
    projection=projection
)

ax2.set_extent(
    [35, 68, 22, 43],
    crs=projection
)

ax2.add_feature(
    cfeature.LAND,
    facecolor="#f3f0e8",
    zorder=0
)

ax2.add_feature(
    cfeature.OCEAN,
    facecolor="#e5f1f7",
    zorder=0
)

ax2.add_feature(
    cfeature.COASTLINE,
    linewidth=0.7
)

ax2.add_feature(
    cfeature.BORDERS,
    linewidth=0.6,
    linestyle=":"
)

# Mesh cells
mesh2 = PolyCollection(
    polygons,
    facecolors="none",
    edgecolors="gray",
    linewidths=0.35,
    transform=projection,
    zorder=2
)

ax2.add_collection(mesh2)

# Boundary cells
boundary_mesh2 = PolyCollection(
    boundary_polygons,
    facecolors="none",
    edgecolors="red",
    linewidths=0.7,
    transform=projection,
    zorder=3
)

ax2.add_collection(boundary_mesh2)

# Cell centers
ax2.scatter(
    lon_cell,
    lat_cell,
    s=2,
    color="black",
    transform=projection,
    zorder=4
)

# Iran reference marker
ax2.scatter(
    53.7,
    32.0,
    marker="*",
    s=100,
    color="darkorange",
    edgecolor="black",
    transform=projection,
    zorder=5,
    label="Iran reference point"
)

ax2.set_title(
    "MPAS Mesh - Iran and Neighboring Regions",
    fontsize=14,
    weight="bold"
)

gl2 = ax2.gridlines(
    draw_labels=True,
    linewidth=0.4,
    color="gray",
    alpha=0.5,
    linestyle="--"
)

gl2.top_labels = False
gl2.right_labels = False

ax2.legend(loc="lower left")


# ==========================================================
# FINAL
# ==========================================================

fig.suptitle(
    "MPAS Limited-Area Mesh Coverage",
    fontsize=17,
    weight="bold"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

print("\nMap saved:")
print(OUTPUT_FILE)

plt.show()
