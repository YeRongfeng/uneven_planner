#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 CANONICAL_MANIFEST OUTPUT_DIR [TRAIN_PATHS] [VAL_PATHS] [PARALLEL_WORKERS] [ROS_PORT]" >&2
echo "Environment overrides: DOMAIN TRAIN_SITE_ID VAL_SITE_ID TRAIN_SOURCE_PROFILE VAL_SOURCE_PROFILE LOG_DIR GENERATION_SEED" >&2
}

if [[ $# -lt 2 || $# -gt 6 ]]; then
    usage
    exit 2
fi

manifest="$1"
output_dir="$(realpath -m "$2")"
train_paths="${3:-100}"
val_paths="${4:-10}"
parallel_workers="${5:-4}"
ros_port="${6:-11412}"

if [[ ! -f "${manifest}" ]]; then
    echo "Missing canonical manifest: ${manifest}" >&2
    exit 1
fi
if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing output: ${output_dir}" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "${script_dir}/../../../.." && pwd)"
map_values=()
while IFS= read -r value; do
    map_values+=("${value}")
done < <(python3 - "${manifest}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
print(",".join(manifest["train_maps"]))
print(",".join(manifest["val_maps"]))
print(manifest.get("domain", ""))
print(manifest.get("train_site_id", ""))
print(manifest.get("val_site_id", ""))
print(manifest.get("train_source_profile", "als"))
print(manifest.get("val_source_profile", "als"))
PY
)
train_map_list="${map_values[0]}"
val_map_list="${map_values[1]}"
manifest_domain="${map_values[2]}"
manifest_train_site_id="${map_values[3]}"
manifest_val_site_id="${map_values[4]}"
manifest_train_source_profile="${map_values[5]}"
manifest_val_source_profile="${map_values[6]}"
train_count="$(printf '%s' "${train_map_list}" | awk -F, '{print NF}')"
val_count="$(printf '%s' "${val_map_list}" | awk -F, '{print NF}')"

if [[ -z "${train_map_list}" || -z "${val_map_list}" ]]; then
    echo "Canonical manifest must contain train_maps and val_maps" >&2
    exit 1
fi

mkdir -p "${output_dir}"
domain="${DOMAIN:-${manifest_domain}}"
train_site_id="${TRAIN_SITE_ID:-${manifest_train_site_id}}"
val_site_id="${VAL_SITE_ID:-${manifest_val_site_id}}"
train_source_profile="${TRAIN_SOURCE_PROFILE:-${manifest_train_source_profile}}"
val_source_profile="${VAL_SOURCE_PROFILE:-${manifest_val_source_profile}}"
log_dir="${LOG_DIR:-${output_dir}.logs}"
generation_seed="${GENERATION_SEED:-20260815}"
if [[ -z "${domain}" || -z "${train_site_id}" || -z "${val_site_id}" ]]; then
    echo "Canonical manifest lacks domain/site metadata; set DOMAIN, TRAIN_SITE_ID, and VAL_SITE_ID" >&2
    exit 1
fi
mkdir -p "${log_dir}"
source "${workspace}/devel/setup.bash"

count_files() {
    local root="$1"
    local pattern="$2"
    if [[ ! -d "${root}" ]]; then
        echo 0
        return
    fi
    find "${root}" -type f -name "${pattern}" -printf . | wc -c
}

worker_failed() {
    rg -q '"status"[[:space:]]*:[[:space:]]*"failed"' \
        "${output_dir}"/experiment_manifest_worker_*.json 2>/dev/null
}

active_pid=""
stop_active_launch() {
    if [[ -n "${active_pid}" ]] && kill -0 "${active_pid}" 2>/dev/null; then
        kill -INT -- "-${active_pid}" 2>/dev/null || true
        wait "${active_pid}" 2>/dev/null || true
    fi
    active_pid=""
}
trap stop_active_launch EXIT INT TERM

run_launch() {
    local phase="$1"
    local environments="$2"
    local start_env="$3"
    local train_pool="$4"
    local val_pool="$5"
    local log_file="${log_dir}/${phase}.log"
    local train_expected val_expected train_paths_expected val_paths_expected

    if [[ "${phase}" == "phase_a" ]]; then
        train_expected="${environments}"
        val_expected="${environments}"
        train_paths_expected=$((environments * train_paths))
        val_paths_expected=$((environments * val_paths))
    else
        train_expected="${train_count}"
        val_expected="${val_count}"
        train_paths_expected=$((train_count * train_paths))
        val_paths_expected=$((val_count * val_paths))
    fi

    local split_args=(
        train_external_source_profile:="${train_source_profile}"
        val_external_source_profile:="${val_source_profile}"
        train_external_site_id:="${train_site_id}"
        val_external_site_id:="${val_site_id}"
    )
    if [[ "${phase}" == "phase_a" ]]; then
        split_args+=(
            train_external_map_paths:="${train_pool}"
            val_external_map_paths:="${val_pool}"
            canonical_primary_scene_count:="${environments}"
            canonical_pool_start_env_id:="${start_env}"
        )
    else
        split_args+=(
            canonical_primary_scene_count:="${environments}"
            canonical_pool_start_env_id:="${start_env}"
        )
    fi

    echo "Starting ${phase}: environments=${environments}, start_env=${start_env}" >&2
    setsid env ROS_MASTER_URI="http://localhost:${ros_port}" \
        ROS_LOG_DIR="${log_dir}/ros_${phase}" \
        roslaunch plan_manager terrain_dataset_generation_parallel.launch \
        parallel_workers:="${parallel_workers}" \
        num_environments:="${environments}" \
        train_paths_per_env:="${train_paths}" \
        val_paths_per_env:="${val_paths}" \
        stop_after_train:="$([[ "${phase}" == "phase_b" ]] && echo true || echo false)" \
        dataset_dir:="${output_dir}" \
        start_env_id:="${start_env}" \
        external_map_path:="${train_map_list}" \
        external_map_paths:="${train_map_list}" \
        external_map_format:=pcd \
        external_domain:="${domain}" \
        external_map_is_canonical:=true \
        canonical_maps_per_environment:=1 \
        target_map_size:=20.0 \
        target_resolution:=0.2 \
        external_map_physical_size:=0.0 \
        external_map_min_physical_size:=0.0 \
        scale_external_map_z:=false \
        external_map_fixed_yaw_deg:=0.0 \
        crop_min_coverage:=0.97 \
        crop_max_attempts:=1 \
        generation_random_seed:="${generation_seed}" \
        max_path_retries_before_regenerate:=12 \
        enable_rviz:=false \
        "${split_args[@]}" >"${log_file}" 2>&1 &
    active_pid=$!

    local poll=0
    while true; do
        poll=$((poll + 1))
        local train_maps val_maps train_saved val_saved
        train_maps="$(count_files "${output_dir}/train" map.p)"
        val_maps="$(count_files "${output_dir}/val" map.p)"
        train_saved="$(count_files "${output_dir}/train" 'path_*.p')"
        val_saved="$(count_files "${output_dir}/val" 'path_*.p')"
        if ((poll == 1 || poll % 4 == 0)); then
            echo "${phase}: maps ${train_maps}/${train_expected} train, ${val_maps}/${val_expected} val; paths ${train_saved}/${train_paths_expected} train, ${val_saved}/${val_paths_expected} val" >&2
        fi
        if [[ "${train_maps}" -eq "${train_expected}" &&
              "${val_maps}" -eq "${val_expected}" &&
              "${train_saved}" -eq "${train_paths_expected}" &&
              "${val_saved}" -eq "${val_paths_expected}" ]]; then
            stop_active_launch
            return 0
        fi
        if worker_failed; then
            echo "${phase} worker reported failed; inspect ${log_file}" >&2
            stop_active_launch
            return 1
        fi
        if ! kill -0 "${active_pid}" 2>/dev/null; then
            echo "${phase} roslaunch exited early; inspect ${log_file}" >&2
            return 1
        fi
        sleep 15
    done
}

val_count_for_phase="${val_count}"
run_launch phase_a "${val_count_for_phase}" 0 "${train_map_list}" "${val_map_list}"

remaining_train=$((train_count - val_count_for_phase))
if ((remaining_train > 0)); then
    run_launch phase_b "${remaining_train}" "${val_count_for_phase}" "${train_map_list}" ""
fi

trap - EXIT INT TERM
echo "Completed approved canonical dataset: ${output_dir}" >&2
