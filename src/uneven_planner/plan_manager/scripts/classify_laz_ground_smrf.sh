#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 INPUT.laz OUTPUT.laz" >&2
    echo "Environment: PDAL_BIN, SMRF_CELL=1.0, SMRF_SCALAR=1.25, SMRF_SLOPE=0.15, SMRF_THRESHOLD=0.5, SMRF_WINDOW=18.0" >&2
}

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

input_laz="$1"
output_laz="$2"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "${script_dir}/../../../.." && pwd)"
pdal_prefix="${workspace}/.conda/pdal"
pdal_bin="${PDAL_BIN:-${pdal_prefix}/bin/pdal}"

if [[ ! -f "${input_laz}" ]]; then
    echo "Missing input: ${input_laz}" >&2
    exit 1
fi
if [[ -e "${output_laz}" ]]; then
    echo "Refusing to overwrite: ${output_laz}" >&2
    exit 1
fi
if [[ ! -x "${pdal_bin}" ]]; then
    echo "PDAL executable not found: ${pdal_bin}" >&2
    exit 1
fi

mkdir -p "$(dirname "${output_laz}")"
export PROJ_DATA="${PROJ_DATA:-${pdal_prefix}/share/proj}"
export GDAL_DATA="${GDAL_DATA:-${pdal_prefix}/share/gdal}"

"${pdal_bin}" translate "${input_laz}" "${output_laz}" \
    assign outlier smrf \
    --filters.assign.value='Classification = 0' \
    --filters.outlier.method=statistical \
    --filters.outlier.mean_k=8 \
    --filters.outlier.multiplier=3.0 \
    --filters.smrf.cell="${SMRF_CELL:-1.0}" \
    --filters.smrf.scalar="${SMRF_SCALAR:-1.25}" \
    --filters.smrf.slope="${SMRF_SLOPE:-0.15}" \
    --filters.smrf.threshold="${SMRF_THRESHOLD:-0.5}" \
    --filters.smrf.window="${SMRF_WINDOW:-18.0}" \
    --filters.smrf.returns='last,only' \
    --writers.las.compression=true \
    --writers.las.minor_version=4 \
    --writers.las.dataformat_id=6

