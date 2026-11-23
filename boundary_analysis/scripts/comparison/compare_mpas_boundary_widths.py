#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare CSV network boundary widths with geometric-normal widths.

Run in the folder containing iran.static.nc and local_boundary_width.csv.
Inputs are read only. Outputs:
  boundary_width_comparison.csv
  boundary_width_comparison_report.txt
  boundary_width_geometry_769.png, ..._777.png, ..._828.png
"""
import os, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from pyproj import CRS, Transformer, Geod

PROJECT_ROOT = Path(__file__).resolve().parents[3]

NC_FILE = PROJECT_ROOT / "iran.static.nc"

INPUT_CSV = (
    PROJECT_ROOT
    / "boundary_analysis"
    / "results"
    / "geometric"
    / "local_boundary_width.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "boundary_analysis"
    / "results"
    / "comparison"
)

OUTPUT_CSV = OUTPUT_DIR / "boundary_width_comparison.csv"
REPORT_FILE = OUTPUT_DIR / "boundary_width_comparison_report.txt"
EXPECTED_INNER = 207
EXPECTED_PHYSICAL = 277
SAMPLE_EDGES = [769, 777, 828]
GEOD = Geod(ellps="WGS84")
EPS = 1e-9
TOL = 1e-8


def degrees(a, units=""):
    a = np.asarray(np.ma.filled(a, np.nan), dtype=float)
    u = str(units).lower()
    if "radian" in u:
        return np.degrees(a)
    if "degree" in u:
        return a
    finite = a[np.isfinite(a)]
    return np.degrees(a) if finite.size and np.nanmax(np.abs(finite)) <= 2*np.pi+0.1 else a


def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]


def normalize(v):
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("Zero-length vector.")
    return v/n


def projection(lon0, lat0):
    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    return (
        Transformer.from_crs("EPSG:4326", crs, always_xy=True),
        Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    )


def ray_hit(p, d, a, b):
    s = b-a
    den = cross(d, s)
    if abs(den) < EPS:
        return None
    delta = a-p
    t = cross(delta, s)/den
    u = cross(delta, d)/den
    if t <= EPS or u < -TOL or u > 1+TOL:
        return None
    return float(t), float(u), p+t*d


def segment(edge, data, fwd):
    v1, v2 = map(int, data["v_edge"][edge])
    if v1 <= 0 or v2 <= 0:
        raise ValueError(f"Invalid vertex IDs on edge {edge}")
    v1 -= 1
    v2 -= 1
    ax, ay = fwd.transform(data["lon_v"][v1], data["lat_v"][v1])
    bx, by = fwd.transform(data["lon_v"][v2], data["lat_v"][v2])
    return np.array([ax, ay]), np.array([bx, by])


def edge_midpoint(edge, data, fwd):
    a, b = segment(edge, data, fwd)
    return (a+b)/2


def geod_km(lon1, lat1, lon2, lat2):
    return GEOD.inv(lon1, lat1, lon2, lat2)[2]/1000


def load():
    df = pd.read_csv(INPUT_CSV)
    needed = {"inner_edge","start_cell","outer_edge","outer_cell",
              "width_km","path_cell_ids","mask_sequence"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {sorted(needed-set(df.columns))}")
    with Dataset(NC_FILE) as nc:
        def coord(name):
            v = nc.variables[name]
            return degrees(v[:], v.getncattr("units") if "units" in v.ncattrs() else "")
        d = {
            "lat_c": coord("latCell"), "lon_c": coord("lonCell"),
            "lat_v": coord("latVertex"), "lon_v": coord("lonVertex"),
            "mask": np.asarray(np.ma.filled(nc.variables["bdyMaskCell"][:], -999), int),
            "c_edge": np.asarray(np.ma.filled(nc.variables["cellsOnEdge"][:], 0), int),
            "v_edge": np.asarray(np.ma.filled(nc.variables["verticesOnEdge"][:], 0), int)
        }
    return df, d


def physical_edges(d):
    out = []
    for e, (c1,c2) in enumerate(d["c_edge"]):
        c1, c2 = int(c1), int(c2)
        if (c1 == 0) == (c2 == 0):
            continue
        c = (c2 if c1 == 0 else c1)-1
        if 0 <= c < len(d["mask"]) and d["mask"][c] == 7:
            out.append(e)
    if len(out) != EXPECTED_PHYSICAL:
        raise ValueError(f"Expected {EXPECTED_PHYSICAL} physical Mask-7 edges; found {len(out)}")
    return out


def calculate(row, d, physical):
    e, start, outer_e, outer_c = [int(row[k]) for k in
                                  ("inner_edge","start_cell","outer_edge","outer_cell")]
    path = [int(x) for x in str(row["path_cell_ids"]).split(";") if x.strip()]
    if not path or path[0] != start or path[-1] != outer_c:
        raise ValueError("CSV path endpoints do not match start_cell/outer_cell")
    if d["mask"][start] != 1 or d["mask"][outer_c] != 7:
        raise ValueError("Start/outer cell mask validation failed")
    if outer_e not in physical:
        raise ValueError(f"CSV outer_edge {outer_e} is not a physical edge")
    adj = [int(x)-1 for x in d["c_edge"][outer_e] if int(x)>0]
    if outer_c not in adj:
        raise ValueError("CSV outer_edge is not adjacent to CSV outer_cell")

    iv1,iv2 = map(int,d["v_edge"][e])
    if iv1 <= 0 or iv2 <= 0:
        raise ValueError("Invalid inner-edge vertices")
    iv1 -= 1; iv2 -= 1
    lon1,lat1 = d["lon_v"][iv1],d["lat_v"][iv1]
    lon2,lat2 = d["lon_v"][iv2],d["lat_v"][iv2]
    lon0 = math.degrees(math.atan2(
        math.sin(math.radians(lon1))+math.sin(math.radians(lon2)),
        math.cos(math.radians(lon1))+math.cos(math.radians(lon2))))
    lat0 = (lat1+lat2)/2
    fwd,inv = projection(lon0,lat0)

    p1 = np.array(fwd.transform(lon1,lat1))
    p2 = np.array(fwd.transform(lon2,lat2))
    origin = (p1+p2)/2
    tangent = normalize(p2-p1)
    normal = np.array([-tangent[1],tangent[0]])

    sx,sy = fwd.transform(d["lon_c"][start],d["lat_c"][start])
    if np.dot(normal,np.array([sx,sy])-origin) < 0:
        normal = -normal
    if np.dot(normal,np.array([sx,sy])-origin) <= 0:
        raise ValueError("Could not orient normal toward Mask 1")

    net_mid = edge_midpoint(outer_e,d,fwd)
    net_lon,net_lat = inv.transform(*net_mid)
    net_chord = np.linalg.norm(net_mid-origin)/1000
    net_geod = geod_km(lon0,lat0,net_lon,net_lat)
    vec = net_mid-origin
    angle = math.degrees(math.acos(np.clip(
        np.dot(normal,vec)/np.linalg.norm(vec),-1,1)))

    hits=[]
    for pe in physical:
        a,b=segment(pe,d,fwd)
        h=ray_hit(origin,normal,a,b)
        if h is not None:
            hits.append((h[0],pe,h[1],h[2]))
    if not hits:
        raise ValueError("No positive intersection with physical Mask-7 boundary")
    dist, normal_e, frac, hit=min(hits,key=lambda x:x[0])
    normal_lon,normal_lat=inv.transform(*hit)
    normal_km=dist/1000
    exit_sep=geod_km(net_lon,net_lat,normal_lon,normal_lat)

    own_a,own_b=segment(outer_e,d,fwd)
    own_hit=ray_hit(origin,normal,own_a,own_b)

    return {
        "inner_edge":e, "start_cell":start, "outer_cell":outer_c,
        "network_outer_edge":outer_e, "normal_outer_edge":normal_e,
        "csv_network_path_width_km":float(row["width_km"]),
        "network_inner_mid_to_outer_edge_mid_km":net_chord,
        "network_midpoint_geodesic_km":net_geod,
        "normal_width_km":normal_km,
        "normal_minus_network_midpoint_km":normal_km-net_chord,
        "normal_minus_csv_path_width_km":normal_km-float(row["width_km"]),
        "normal_vs_network_vector_angle_deg":angle,
        "boundary_exit_separation_km":exit_sep,
        "network_exit_lon":net_lon,"network_exit_lat":net_lat,
        "normal_exit_lon":normal_lon,"normal_exit_lat":normal_lat,
        "normal_edge_fraction":frac,
        "normal_intersects_csv_outer_edge":own_hit is not None,
        "normal_to_csv_outer_edge_hit_km":own_hit[0]/1000 if own_hit else np.nan,
        "path_cell_ids":";".join(map(str,path)),
        "mask_sequence":str(row["mask_sequence"]),
        "status":"SUCCESS"
    }


def plot_sample(row, result, d):
    e=int(row["inner_edge"])
    ids=[int(x) for x in str(row["path_cell_ids"]).split(";") if x.strip()]
    lonp,latp=d["lon_c"][ids],d["lat_c"][ids]
    iv=d["v_edge"][e]-1
    ov=d["v_edge"][int(row["outer_edge"])]-1
    fig,ax=plt.subplots(figsize=(10,8))
    ax.plot(lonp,latp,"-o",lw=2,label="Network cell-center path")
    ax.plot(d["lon_v"][iv],d["lat_v"][iv],"k-",lw=4,label="Inner edge")
    ax.plot(d["lon_v"][ov],d["lat_v"][ov],"b-",lw=4,label="CSV outer edge")
    # Use the geographic midpoint of the inner edge for the displayed ray origin.
    lon0=float(np.mean(d["lon_v"][iv])); lat0=float(np.mean(d["lat_v"][iv]))
    ax.plot([lon0,result["normal_exit_lon"]],
            [lat0,result["normal_exit_lat"]],"r--",lw=2,label="Geometric normal")
    ax.scatter([result["network_exit_lon"]],[result["network_exit_lat"]],
               marker="s",s=90,label="Network exit midpoint")
    ax.scatter([result["normal_exit_lon"]],[result["normal_exit_lat"]],
               marker="*",s=150,label="Normal exit")
    ax.set_title(
        f"Edge {e}: normal {result['normal_width_km']:.1f} km | "
        f"network chord {result['network_inner_mid_to_outer_edge_mid_km']:.1f} km | "
        f"exit separation {result['boundary_exit_separation_km']:.1f} km")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True,alpha=.3); ax.legend(); ax.set_aspect("equal",adjustable="datalim")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"boundary_width_geometry_{e}.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df,d=load()
    print(f"CSV rows: {len(df)} | cells: {len(d['mask'])} | edges: {len(d['c_edge'])}")
    if len(df)!=EXPECTED_INNER:
        raise ValueError(f"Expected {EXPECTED_INNER} rows, found {len(df)}")
    physical=physical_edges(d)
    print(f"Physical Mask-7 edges: {len(physical)}")

    results=[]; errors=[]
    for i,row in df.iterrows():
        try:
            results.append(calculate(row,d,physical))
        except Exception as exc:
            errors.append((int(row["inner_edge"]),str(exc)))
        if (i+1)%25==0 or i+1==len(df):
            print(f"Processed {i+1}/{len(df)}")

    out=pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV,index=False,float_format="%.6f")

    for e in SAMPLE_EDGES:
        rr=df.loc[df.inner_edge==e]
        oo=out.loc[out.inner_edge==e]
        if not rr.empty and not oo.empty:
            plot_sample(rr.iloc[0],oo.iloc[0],d)

    lines=[
        "MPAS NETWORK / GEOMETRIC NORMAL COMPARISON",
        "="*70,
        f"Input rows: {len(df)}",
        f"Successful: {len(out)}",
        f"Failed: {len(errors)}",
        f"Physical Mask-7 edges: {len(physical)}",
        "",
        "SUMMARY STATISTICS"
    ]
    cols=[
        "csv_network_path_width_km",
        "network_inner_mid_to_outer_edge_mid_km",
        "normal_width_km",
        "normal_minus_network_midpoint_km",
        "normal_minus_csv_path_width_km",
        "boundary_exit_separation_km",
        "normal_vs_network_vector_angle_deg"
    ]
    for c in cols:
        s=out[c].dropna()
        if len(s):
            lines.append(f"{c}: n={len(s)}, mean={s.mean():.3f}, "
                         f"median={s.median():.3f}, min={s.min():.3f}, max={s.max():.3f}")
    if len(out):
        lines += ["", "Normal ray intersects CSV outer edge: "
                  f"{int(out.normal_intersects_csv_outer_edge.sum())}/{len(out)}"]
        lines += ["", "SAMPLE EDGES"]
        for e in SAMPLE_EDGES:
            s=out.loc[out.inner_edge==e]
            if len(s):
                r=s.iloc[0]
                lines.append(
                    f"Edge {e}: network edge={r.network_outer_edge}, normal edge={r.normal_outer_edge}, "
                    f"network chord={r.network_inner_mid_to_outer_edge_mid_km:.3f} km, "
                    f"normal={r.normal_width_km:.3f} km, "
                    f"exit separation={r.boundary_exit_separation_km:.3f} km, "
                    f"angle={r.normal_vs_network_vector_angle_deg:.2f} deg, "
                    f"ray intersects CSV edge={r.normal_intersects_csv_outer_edge}")
    if errors:
        lines += ["", "FAILED EDGES"]
        lines += [f"Edge {e}: {msg}" for e,msg in errors]

    report="\n".join(lines)
    with open(REPORT_FILE,"w",encoding="utf-8") as f:
        f.write(report+"\n")
    print("\n"+report)
    print("\nCreated:",OUTPUT_CSV,REPORT_FILE)
    for e in SAMPLE_EDGES:
        p=f"boundary_width_geometry_{e}.png"
        if os.path.exists(p): print("Created:",p)


if __name__=="__main__":
    main()
