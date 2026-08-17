#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 TRAIN.laz VAL.laz OUTPUT_DIR [SCENES [TRAIN_PATHS [VAL_PATHS]]]" >&2
    echo "Environment: SEED=20260813 CROP_MIN_COVERAGE=0.97 CROP_MAX_ATTEMPTS=50 ACCEPTED_GRADES=easy,medium,hard PARALLEL_WORKERS=1" >&2
    echo "             ABOVE_SURFACE_HEIGHT_M=2.0 ABOVE_SURFACE_CELL_SIZE_M=1.0 MAX_ABOVE_SURFACE_COVERAGE=1.0 MIN_ABOVE_SURFACE_COMPONENT_CELLS=0" >&2
    echo "             TREE_CLASS_VALUES=1 GROUND_CLASS_VALUES=2 MIN_TREE_COMPONENT_CELLS=0" >&2
    echo "             TRAIN_SOURCE_PROFILE=uls VAL_SOURCE_PROFILE=uls" >&2
    echo "             DOMAIN=... TRAIN_SITE_ID=... VAL_SITE_ID=... TRAIN_SOURCE_URL=... VAL_SOURCE_URL=..." >&2
    echo "             TRAIN_LICENSE=... VAL_LICENSE=... TRAIN_CRS=... VAL_CRS=..." >&2
}

if [[ $# -lt 3 || $# -gt 6 ]]; then
    usage
    exit 2
fi

train_laz="$1"
val_laz="$2"
output_dir="$3"
scenes="${4:-1}"
train_paths="${5:-2}"
val_paths="${6:-1}"

seed="${SEED:-20260813}"
accepted_grades="${ACCEPTED_GRADES:-easy,medium,hard}"
parallel_workers="${PARALLEL_WORKERS:-1}"
crop_min_coverage="${CROP_MIN_COVERAGE:-0.97}"
crop_max_attempts="${CROP_MAX_ATTEMPTS:-50}"
above_surface_height_m="${ABOVE_SURFACE_HEIGHT_M:-2.0}"
above_surface_cell_size_m="${ABOVE_SURFACE_CELL_SIZE_M:-1.0}"
max_above_surface_coverage="${MAX_ABOVE_SURFACE_COVERAGE:-1.0}"
min_above_surface_component_cells="${MIN_ABOVE_SURFACE_COMPONENT_CELLS:-0}"
tree_class_values="${TREE_CLASS_VALUES:-1}"
ground_class_values="${GROUND_CLASS_VALUES:-2}"
min_tree_component_cells="${MIN_TREE_COMPONENT_CELLS:-0}"
poll_seconds="${POLL_SECONDS:-2}"
# Full 100-scene releases can legitimately need several hours on difficult
# terrain.  Keep the polling interval short but allow an eight-hour wall-clock
# window by default; callers may still set a tighter test timeout explicitly.
max_polls="${MAX_PIPELINE_POLLS:-14400}"
resume_generation="${RESUME_GENERATION:-0}"
train_source_profile="${TRAIN_SOURCE_PROFILE:-uls}"
val_source_profile="${VAL_SOURCE_PROFILE:-uls}"
train_source_url="${TRAIN_SOURCE_URL:-}"
val_source_url="${VAL_SOURCE_URL:-}"
train_license="${TRAIN_LICENSE:-}"
val_license="${VAL_LICENSE:-}"
train_crs="${TRAIN_CRS:-}"
val_crs="${VAL_CRS:-}"
domain="${DOMAIN:-}"
train_site_id="${TRAIN_SITE_ID:-}"
val_site_id="${VAL_SITE_ID:-}"

if [[ -n "${train_site_id}" && "${train_site_id}" == "${val_site_id}" ]]; then
    echo "Train/val leakage: TRAIN_SITE_ID and VAL_SITE_ID are both ${train_site_id}" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "${script_dir}/../../../.." && pwd)"

for source_path in "${train_laz}" "${val_laz}"; do
    if [[ ! -f "${source_path}" ]]; then
        echo "Missing mother map: ${source_path}" >&2
        exit 1
    fi
done
# roslaunch may change its working directory.  Resolve every externally visible
# path before launching so sampling, ROS output, and the completion monitor all
# refer to the same files.
train_laz="$(realpath "${train_laz}")"
val_laz="$(realpath "${val_laz}")"
output_dir="$(realpath -m "${output_dir}")"
if [[ "${resume_generation}" == "1" ]]; then
    if [[ ! -d "${output_dir}" ]]; then
        echo "Resume requested but output is missing: ${output_dir}" >&2
        exit 1
    fi
elif [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing output: ${output_dir}" >&2
    exit 1
fi
trajectory_dir="${output_dir}"
ros_log_dir="${output_dir}.ros_logs"

mkdir -p "${trajectory_dir}" "${ros_log_dir}"
export ROS_LOG_DIR="${ros_log_dir}"

source "${workspace}/devel/setup.bash"
setsid roslaunch plan_manager terrain_dataset_generation_parallel.launch \
    parallel_workers:="${parallel_workers}" \
    num_environments:="${scenes}" \
    train_paths_per_env:="${train_paths}" \
    val_paths_per_env:="${val_paths}" \
    dataset_dir:="${trajectory_dir}" \
    start_env_id:=0 \
    external_map_path:="${train_laz}" \
    external_map_paths:="${train_laz},${val_laz}" \
    train_external_map_paths:="${train_laz}" \
    val_external_map_paths:="${val_laz}" \
    external_map_is_canonical:=false \
    external_map_format:=laz \
    train_external_source_profile:="${train_source_profile}" \
    val_external_source_profile:="${val_source_profile}" \
    external_domain:="${domain}" \
    train_external_source_url:="${train_source_url}" \
    val_external_source_url:="${val_source_url}" \
    train_external_license:="${train_license}" \
    val_external_license:="${val_license}" \
    train_external_crs:="${train_crs}" \
    val_external_crs:="${val_crs}" \
    train_external_site_id:="${train_site_id}" \
    val_external_site_id:="${val_site_id}" \
    external_accepted_grades:="${accepted_grades}" \
    external_above_surface_height:="${above_surface_height_m}" \
    external_above_surface_cell_size:="${above_surface_cell_size_m}" \
    external_max_above_surface_coverage:="${max_above_surface_coverage}" \
    external_min_above_surface_component_cells:="${min_above_surface_component_cells}" \
    external_tree_class_values:="${tree_class_values}" \
    external_ground_class_values:="${ground_class_values}" \
    external_min_tree_component_cells:="${min_tree_component_cells}" \
    target_map_size:=20.0 \
    target_resolution:=0.2 \
    crop_min_coverage:="${crop_min_coverage}" \
    crop_max_attempts:="${crop_max_attempts}" \
    crop_random_seed:="${seed}" \
    generation_random_seed:="$((seed + 2))" \
    max_path_retries_before_regenerate:=12 &
launch_pid=$!

stop_launch() {
    if kill -0 "${launch_pid}" 2>/dev/null; then
        kill -INT -- "-${launch_pid}" 2>/dev/null || true
        wait "${launch_pid}" 2>/dev/null || true
    fi
}
trap stop_launch EXIT INT TERM

expected_train_paths=$((scenes * train_paths))
expected_val_paths=$((scenes * val_paths))
expected_workers="${parallel_workers}"
if (( expected_workers > scenes )); then
    expected_workers="${scenes}"
fi

count_outputs() {
    local root="$1"
    local pattern="$2"
    if [[ ! -d "${root}" ]]; then
        echo 0
        return
    fi
    find "${root}" -type f -name "${pattern}" -printf . | wc -c
}

count_completed_manifests() {
    local manifest count=0
    for manifest in "${trajectory_dir}"/experiment_manifest_worker_*.json; do
        [[ -f "${manifest}" ]] || continue
        if grep -q '"status": "completed"' "${manifest}"; then
            count=$((count + 1))
        fi
    done
    echo "${count}"
}

count_failed_manifests() {
    local manifest count=0
    for manifest in "${trajectory_dir}"/experiment_manifest_worker_*.json; do
        [[ -f "${manifest}" ]] || continue
        if grep -q '"status": "failed"' "${manifest}"; then
            count=$((count + 1))
        fi
    done
    echo "${count}"
}

for ((poll=1; poll<=max_polls; poll++)); do
    train_map_count="$(count_outputs "${trajectory_dir}/train" map.p)"
    val_map_count="$(count_outputs "${trajectory_dir}/val" map.p)"
    train_path_count="$(count_outputs "${trajectory_dir}/train" 'path_*.p')"
    val_path_count="$(count_outputs "${trajectory_dir}/val" 'path_*.p')"
    completed_manifest_count="$(count_completed_manifests)"
    failed_manifest_count="$(count_failed_manifests)"

    if [[ "${failed_manifest_count}" -gt 0 ]]; then
        stop_launch
        trap - EXIT INT TERM
        echo "ROS generation stopped after a worker reported failure: ${failed_manifest_count} worker(s)" >&2
        exit 1
    fi

    if [[ "${train_map_count}" -eq "${scenes}" &&
          "${val_map_count}" -eq "${scenes}" &&
          "${train_path_count}" -eq "${expected_train_paths}" &&
          "${val_path_count}" -eq "${expected_val_paths}" &&
          "${completed_manifest_count}" -eq "${expected_workers}" ]]; then
        stop_launch
        trap - EXIT INT TERM
        echo "Completed: ${scenes} train/val scenes, ${train_path_count} train paths, ${val_path_count} val paths"
        exit 0
    fi
    if ! kill -0 "${launch_pid}" 2>/dev/null; then
        wait "${launch_pid}" || true
        echo "ROS generation exited before reaching the requested output counts" >&2
        exit 1
    fi
    sleep "${poll_seconds}"
done

echo "Timed out before reaching the requested output counts" >&2
exit 1
