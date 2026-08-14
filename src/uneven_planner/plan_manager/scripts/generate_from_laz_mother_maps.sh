#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 TRAIN.laz VAL.laz OUTPUT_DIR [SCENES [TRAIN_PATHS [VAL_PATHS]]]" >&2
    echo "Environment: SEED=20260813 MAX_ATTEMPTS_PER_SPLIT=<default:30 per sampled candidate> ACCEPTED_GRADES=easy,medium,hard PARALLEL_WORKERS=1 CANONICAL_CANDIDATES_PER_ENV=2" >&2
    echo "             TRAIN_SOURCE_PROFILE=uls VAL_SOURCE_PROFILE=uls TRAIN_POINT_CLASSES=2 VAL_POINT_CLASSES=2" >&2
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
canonical_candidates="${CANONICAL_CANDIDATES_PER_ENV:-2}"
poll_seconds="${POLL_SECONDS:-2}"
# Full 100-scene releases can legitimately need several hours on difficult
# terrain.  Keep the polling interval short but allow an eight-hour wall-clock
# window by default; callers may still set a tighter test timeout explicitly.
max_polls="${MAX_PIPELINE_POLLS:-14400}"
train_source_profile="${TRAIN_SOURCE_PROFILE:-uls}"
val_source_profile="${VAL_SOURCE_PROFILE:-uls}"
train_point_classes="${TRAIN_POINT_CLASSES:-2}"
val_point_classes="${VAL_POINT_CLASSES:-2}"
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
if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing output: ${output_dir}" >&2
    exit 1
fi

# roslaunch may change its working directory.  Resolve every externally visible
# path before launching so sampling, ROS output, and the completion monitor all
# refer to the same files.
train_laz="$(realpath "${train_laz}")"
val_laz="$(realpath "${val_laz}")"
output_dir="$(realpath -m "${output_dir}")"
scene_root="${output_dir}/sampled_scenes"
train_scene_dir="${scene_root}/train"
val_scene_dir="${scene_root}/val"
trajectory_dir="${output_dir}/trajectories"

mkdir -p "${train_scene_dir}" "${val_scene_dir}" "${trajectory_dir}"
export MPLCONFIGDIR="${output_dir}/matplotlib"
export ROS_LOG_DIR="${output_dir}/ros_logs"
mkdir -p "${MPLCONFIGDIR}" "${ROS_LOG_DIR}"

sampled_scenes=$((scenes * canonical_candidates))
max_attempts="${MAX_ATTEMPTS_PER_SPLIT:-$((sampled_scenes * 30))}"

python3 "${script_dir}/sample_laz_mother_map.py" \
    "${train_laz}" "${train_scene_dir}" \
    --accepted "${sampled_scenes}" \
    --max-attempts "${max_attempts}" \
    --seed "${seed}" \
    --accepted-grades "${accepted_grades}" \
    --source-profile "${train_source_profile}" \
    --point-classes "${train_point_classes}" \
    --source-url "${train_source_url}" \
    --license "${train_license}" \
    --crs "${train_crs}" \
    --site-id "${train_site_id}" \
    --domain "${domain}"

python3 "${script_dir}/sample_laz_mother_map.py" \
    "${val_laz}" "${val_scene_dir}" \
    --accepted "${sampled_scenes}" \
    --max-attempts "${max_attempts}" \
    --seed "$((seed + 1))" \
    --accepted-grades "${accepted_grades}" \
    --source-profile "${val_source_profile}" \
    --point-classes "${val_point_classes}" \
    --source-url "${val_source_url}" \
    --license "${val_license}" \
    --crs "${val_crs}" \
    --site-id "${val_site_id}" \
    --domain "${domain}"

train_maps="$(find "${train_scene_dir}" -maxdepth 1 -type f -name 'scene_*.pcd' ! -name '*_grid.pcd' -exec realpath {} \; | sort | paste -sd, -)"
val_maps="$(find "${val_scene_dir}" -maxdepth 1 -type f -name 'scene_*.pcd' ! -name '*_grid.pcd' -exec realpath {} \; | sort | paste -sd, -)"
if [[ -z "${train_maps}" || -z "${val_maps}" ]]; then
    echo "Sampling completed without a usable train/val map list" >&2
    exit 1
fi

source "${workspace}/devel/setup.bash"
setsid roslaunch plan_manager terrain_dataset_generation_parallel.launch \
    parallel_workers:="${parallel_workers}" \
    num_environments:="${scenes}" \
    train_paths_per_env:="${train_paths}" \
    val_paths_per_env:="${val_paths}" \
    dataset_dir:="${trajectory_dir}" \
    start_env_id:=0 \
    external_map_path:="${train_maps%%,*}" \
    external_map_paths:="${train_maps},${val_maps}" \
    train_external_map_paths:="${train_maps}" \
    val_external_map_paths:="${val_maps}" \
    external_map_is_canonical:=true \
    canonical_maps_per_environment:="${canonical_candidates}" \
    canonical_primary_scene_count:="${scenes}" \
    canonical_pool_start_env_id:=0 \
    external_map_format:=pcd \
    target_map_size:=20.0 \
    target_resolution:=0.2 \
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

for ((poll=1; poll<=max_polls; poll++)); do
    train_map_count="$(count_outputs "${trajectory_dir}/train" map.p)"
    val_map_count="$(count_outputs "${trajectory_dir}/val" map.p)"
    train_path_count="$(count_outputs "${trajectory_dir}/train" 'path_*.p')"
    val_path_count="$(count_outputs "${trajectory_dir}/val" 'path_*.p')"
    completed_manifest_count="$(count_completed_manifests)"

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
