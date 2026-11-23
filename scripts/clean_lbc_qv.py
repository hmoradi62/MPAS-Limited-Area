import os
import glob
import shutil
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = str(PROJECT_ROOT / "run_lbc_20261031_00")
OUTPUT_DIR = str(Path(SOURCE_DIR) / "lbc_qv_clean")

files = sorted(glob.glob(os.path.join(SOURCE_DIR, "lbc.*.nc")))

if len(files) != 17:
    raise RuntimeError(f"Expected 17 files, found {len(files)}")

if os.path.exists(OUTPUT_DIR):
    raise RuntimeError(f"Output directory already exists: {OUTPUT_DIR}")

# First pass: validate every source file without modifying anything.
total_negative = 0

for path in files:
    with Dataset(path, "r") as nc:
        if "lbc_qv" not in nc.variables:
            raise RuntimeError(f"Missing lbc_qv in {path}")

        qv = np.ma.asarray(nc.variables["lbc_qv"][:])
        valid = ~np.ma.getmaskarray(qv)
        values = qv.data[valid]

        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"NaN or Inf detected in {path}")

        total_negative += int(np.count_nonzero(values < 0.0))

if total_negative != 8:
    raise RuntimeError(
        f"Expected exactly 8 negative values, found {total_negative}"
    )

print("Source validation passed.")
print("Files:", len(files))
print("Negative QV values:", total_negative)

# Second pass: copy and correct only the copies.
os.makedirs(OUTPUT_DIR)

fixed_total = 0
max_correction = 0.0

for src in files:
    dst = os.path.join(OUTPUT_DIR, os.path.basename(src))
    shutil.copy2(src, dst)

    with Dataset(dst, "r+") as nc:
        var = nc.variables["lbc_qv"]
        qv = np.ma.asarray(var[:]).copy()

        valid = ~np.ma.getmaskarray(qv)
        negative = valid & (qv.data < 0.0)

        count = int(np.count_nonzero(negative))

        if count:
            old_values = qv.data[negative].copy()
            correction = float(np.max(np.abs(old_values)))

            qv.data[negative] = 0.0
            var[:] = qv

            fixed_total += count
            max_correction = max(max_correction, correction)

            print(os.path.basename(src), "corrected:", count,
                  "max change:", correction)

# Final verification of the copied files.
remaining_negative = 0

for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "lbc.*.nc"))):
    with Dataset(path, "r") as nc:
        qv = np.ma.asarray(nc.variables["lbc_qv"][:])
        valid = ~np.ma.getmaskarray(qv)
        values = qv.data[valid]

        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Non-finite value in output: {path}")

        remaining_negative += int(np.count_nonzero(values < 0.0))

if fixed_total != 8 or remaining_negative != 0:
    raise RuntimeError("Final verification failed.")

print("\nCLEANING COMPLETED")
print("Files copied:", len(files))
print("Values corrected:", fixed_total)
print("Remaining negative values:", remaining_negative)
print("Maximum correction:", max_correction)
print("Output:", OUTPUT_DIR)
