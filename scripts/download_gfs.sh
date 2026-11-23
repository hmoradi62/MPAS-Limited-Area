#!/usr/bin/env bash
set -euo pipefail

# Resolve project root independently of the current directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

BASE="https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# Load project configuration
source "$PROJECT_ROOT/config/gfs.env"

# Resolve relative output paths from project root
if [[ "$OUTDIR" != /* ]]; then
    OUTDIR="$PROJECT_ROOT/$OUTDIR"
fi

[[ "$CYCLE" =~ ^[0-9]{8}$ ]] || { echo "Invalid CYCLE"; exit 1; }
[[ "$CYCLE_HOUR" =~ ^(00|06|12|18)$ ]] || { echo "Invalid CYCLE_HOUR"; exit 1; }

for value in "$START_FH" "$END_FH" "$STEP_H"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "Invalid forecast hours"; exit 1; }
done

(( STEP_H > 0 && END_FH >= START_FH )) || {
    echo "Invalid forecast range"
    exit 1
}

command -v curl >/dev/null || { echo "curl not found"; exit 1; }
command -v grib_ls >/dev/null || { echo "grib_ls not found"; exit 1; }

awk -v l="$LEFTLON" -v r="$RIGHTLON" \
    -v t="$TOPLAT" -v b="$BOTTOMLAT" \
    'BEGIN {
        exit !(l >= 0 && r <= 360 && l < r &&
               b >= -90 && t <= 90 && b < t)
    }' || {
    echo "Invalid geographic bounds"
    exit 1
}

validate_grib() {
    local file="$1"
    local report="$2"

    [[ -s "$file" ]] || return 1

    grib_ls "$file" > "$report" 2>&1 || return 1

    awk '
    $2 == "of" && $1 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ {
        if ($4 == "messages") {
            file_read=$1
            file_total=$3
            file_found=1
        }
        if ($4 == "total" && $5 == "messages") {
            total_read=$1
            total_total=$3
            total_found=1
        }
    }
    END {
        exit !(file_found && total_found &&
               file_read > 0 &&
               file_read == file_total &&
               total_read == total_total &&
               total_read == file_read)
    }' "$report"
}

mkdir -p "$OUTDIR"

TMP=""
REPORT=""

cleanup() {
    [[ -z "$TMP" ]] || rm -f -- "$TMP"
    [[ -z "$REPORT" ]] || rm -f -- "$REPORT"
}
trap cleanup EXIT

echo "GFS cycle: ${CYCLE} ${CYCLE_HOUR} UTC"
echo "Forecast: ${START_FH}-${END_FH}, step ${STEP_H}"
echo "Output: $OUTDIR"

for ((H=START_FH; H<=END_FH; H+=STEP_H)); do

    F=$(printf "%03d" "$H")
    NAME="gfs.t${CYCLE_HOUR}z.pgrb2.0p25.f${F}"

    OUT="${OUTDIR}/${NAME}"
    TMP="${OUT}.tmp"
    REPORT="${OUT}.validation.log"

    if [[ -e "$OUT" ]]; then
        if validate_grib "$OUT" "$REPORT"; then
            echo "VALID, skipping: $NAME"
            continue
        else
            echo "ERROR: Existing file is invalid: $OUT"
            echo "Original file preserved."
            exit 1
        fi
    fi

    URL="${BASE}?file=${NAME}&all_var=on&all_lev=on&subregion=&leftlon=${LEFTLON}&rightlon=${RIGHTLON}&toplat=${TOPLAT}&bottomlat=${BOTTOMLAT}&dir=%2Fgfs.${CYCLE}%2F${CYCLE_HOUR}%2Fatmos"

    rm -f -- "$TMP" "$REPORT"

    echo "Downloading: $NAME"

    if ! curl -fL \
        --connect-timeout 20 \
        --max-time 600 \
        --retry 3 \
        --retry-delay 3 \
        "$URL" -o "$TMP"; then

        echo "ERROR: Download failed: $NAME"
        exit 1
    fi

    if ! validate_grib "$TMP" "$REPORT"; then
        echo "ERROR: GRIB validation failed: $NAME"
        echo "Temporary file will be discarded."
        exit 1
    fi

    mv -- "$TMP" "$OUT"

    echo "VALID and saved: $OUT"
    stat -c 'Size: %s bytes' "$OUT"
done

echo "GFS download completed successfully."
