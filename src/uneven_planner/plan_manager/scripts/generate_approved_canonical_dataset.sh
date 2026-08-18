#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 CANONICAL_MANIFEST OUTPUT_DIR [TRAIN_PATHS] [VAL_PATHS] [PARALLEL_WORKERS] [ROS_PORT]" >&2
echo "Environment overrides: DOMAIN TRAIN_SITE_ID VAL_SITE_ID TRAIN_SOURCE_PROFILE VAL_SOURCE_PROFILE LOG_DIR GENERATION_SEED ONLY_FILLED_SLOTS TEST_GENERATION ALLOW_EXISTING" >&2
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
only_filled_slots="${ONLY_FILLED_SLOTS:-0}"
test_generation="${TEST_GENERATION:-0}"
allow_existing="${ALLOW_EXISTING:-0}"
if [[ "${test_generation}" == "1" ]]; then
    train_paths=1
    val_paths=1
    allow_existing=1
fi
if [[ "${only_filled_slots}" == "1" ]]; then
    allow_existing=1
fi
if [[ -e "${output_dir}" && "${allow_existing}" != "1" ]]; then
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

env_is_done() {
    local dir="$1"
    local expected_paths="$2"
    if [[ -f "${dir}/needs_return.json" ]]; then
        return 0
    fi
    if [[ ! -d "${dir}" ]]; then
        return 1
    fi
    local n
    n="$(find "${dir}" -maxdepth 1 -type f -name 'path_*.p' -printf . | wc -c)"
    [[ "${n}" -eq "${expected_paths}" ]]
}

range_is_done() {
    local start_env="$1"
    local env_count="$2"
    local want_train="$3"
    local want_val="$4"
    local index env_id name
    for ((index = 0; index < env_count; index++)); do
        env_id=$((start_env + index))
        printf -v name 'env%06d' "${env_id}"
        if [[ "${want_train}" == "1" ]] && ! env_is_done "${output_dir}/train/${name}" "${train_paths}"; then
            return 1
        fi
        if [[ "${want_val}" == "1" ]] && ! env_is_done "${output_dir}/val/${name}" "${val_paths}"; then
            return 1
        fi
    done
    return 0
}

count_skip_markers() {
    local root="$1"
    if [[ ! -d "${root}" ]]; then
        echo 0
        return
    fi
    find "${root}" -mindepth 2 -maxdepth 2 -type f -name needs_return.json | wc -l
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

    local start_phase="train"
    local stop_after="false"
    local want_train="1"
    local want_val="1"
    if [[ "${phase}" == "phase_b" || "${phase}" == *_train ]]; then
        stop_after="true"
        want_val="0"
    fi
    if [[ "${phase}" == *_val ]]; then
        start_phase="val"
        want_train="0"
    fi

    echo "Starting ${phase}: environments=${environments}, start_env=${start_env}, start_phase=${start_phase}" >&2
    setsid env ROS_MASTER_URI="http://localhost:${ros_port}" \
        ROS_LOG_DIR="${log_dir}/ros_${phase}" \
        roslaunch plan_manager terrain_dataset_generation_parallel.launch \
        parallel_workers:="${parallel_workers}" \
        num_environments:="${environments}" \
        train_paths_per_env:="${train_paths}" \
        val_paths_per_env:="${val_paths}" \
        stop_after_train:="${stop_after}" \
        start_phase:="${start_phase}" \
        dataset_dir:="${output_dir}" \
        start_env_id:="${start_env}" \
        external_map_path:="${train_map_list}" \
        external_map_paths:="${train_map_list}" \
        external_map_format:=pcd \
        external_domain:="${domain}" \
        external_map_is_canonical:=true \
        canonical_maps_per_environment:=1 \
        target_map_size:=20.0 \
        target_resolution:=0.1 \
        external_map_physical_size:=0.0 \
        external_map_min_physical_size:=0.0 \
        scale_external_map_z:=false \
        external_map_fixed_yaw_deg:=0.0 \
        crop_min_coverage:=0.97 \
        crop_max_attempts:=1 \
        generation_random_seed:="${generation_seed}" \
        max_path_retries_before_regenerate:=30 \
        mark_unplannable_canonical:="$([[ "${test_generation}" == "1" ]] && echo true || echo false)" \
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
        local train_skipped val_skipped
        train_skipped="$(count_skip_markers "${output_dir}/train")"
        val_skipped="$(count_skip_markers "${output_dir}/val")"
        if ((poll == 1 || poll % 4 == 0)); then
            echo "${phase}: maps ${train_maps}/${train_expected} train, ${val_maps}/${val_expected} val; paths ${train_saved}/${train_paths_expected} train, ${val_saved}/${val_paths_expected} val; skip ${train_skipped} train, ${val_skipped} val" >&2
        fi
        if range_is_done "${start_env}" "${environments}" "${want_train}" "${want_val}"; then
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

nth_csv() {
    local csv="$1"
    local index="$2"
    python3 - "${csv}" "${index}" <<'PY'
import sys
values = sys.argv[1].split(",")
index = int(sys.argv[2])
if index < 0 or index >= len(values):
    raise SystemExit(f"map index {index} is outside the canonical list")
print(values[index])
PY
}

if [[ "${only_filled_slots}" == "1" ]]; then
    filled_values=()
    while IFS= read -r value; do
        filled_values+=("${value}")
    done < <(python3 - "${manifest}" "${script_dir}" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2])
from review_slots import filled_records, resolve_slots

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
review = manifest.get("review_file")
if not review:
    raise SystemExit("manifest has no review_file")
replacement = manifest.get("replacement_file") or str(
    Path(review).with_name("canonical_replacements.jsonl"))
state = resolve_slots(review, replacement)
train = sorted({key[1] for key, _, _ in filled_records(state) if key[0] == "train"})
val = sorted({key[1] for key, _, _ in filled_records(state) if key[0] == "val"})
print(",".join(str(index) for index in train))
print(",".join(str(index) for index in val))
PY
    )
    filled_train="${filled_values[0]}"
    filled_val="${filled_values[1]}"
    if [[ -z "${filled_train}" && -z "${filled_val}" ]]; then
        echo "没有已补位的地图可生成最终轨迹" >&2
        exit 1
    fi
    declare -A do_train=()
    declare -A do_val=()
    if [[ -n "${filled_train}" ]]; then
        IFS=',' read -ra train_ids <<< "${filled_train}"
        for index in "${train_ids[@]}"; do
            do_train["${index}"]=1
        done
    fi
    if [[ -n "${filled_val}" ]]; then
        IFS=',' read -ra val_ids <<< "${filled_val}"
        for index in "${val_ids[@]}"; do
            do_val["${index}"]=1
        done
    fi
    unique_ids=()
    index_keys=()
    ((${#do_train[@]})) && index_keys+=("${!do_train[@]}")
    ((${#do_val[@]})) && index_keys+=("${!do_val[@]}")
    while IFS= read -r index; do
        [[ -n "${index}" ]] && unique_ids+=("${index}")
    done < <(printf '%s\n' "${index_keys[@]}" | sort -n | uniq)
    echo "Only-filled trajectory generation for map indices: ${unique_ids[*]}" >&2
    for index in "${unique_ids[@]}"; do
        train_one=""
        val_one=""
        phase_name="fill_${index}"
        if [[ -n "${do_train[${index}]:-}" ]]; then
            train_one="$(nth_csv "${train_map_list}" "${index}")"
        fi
        if [[ -n "${do_val[${index}]:-}" ]]; then
            val_one="$(nth_csv "${val_map_list}" "${index}")"
        fi
        if [[ -n "${train_one}" && -z "${val_one}" ]]; then
            phase_name="fill_${index}_train"
            val_one="${train_one}"
        elif [[ -z "${train_one}" && -n "${val_one}" ]]; then
            phase_name="fill_${index}_val"
            train_one="${val_one}"
        fi
        run_launch "${phase_name}" 1 "${index}" "${train_one}" "${val_one}"
    done
else
    val_count_for_phase="${val_count}"
    run_launch phase_a "${val_count_for_phase}" 0 "${train_map_list}" "${val_map_list}"

    remaining_train=$((train_count - val_count_for_phase))
    if ((remaining_train > 0)); then
        run_launch phase_b "${remaining_train}" "${val_count_for_phase}" "${train_map_list}" ""
    fi
fi

trap - EXIT INT TERM
if [[ "${test_generation}" == "1" ]]; then
    review_file="$(python3 - "${manifest}" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(manifest.get("review_file") or "")
PY
)"
    if [[ -n "${review_file}" ]]; then
        python3 - "${output_dir}" "${review_file}" "${script_dir}" <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from review_slots import append_returns_for_dataset
result = append_returns_for_dataset(sys.argv[1], sys.argv[2])
print(json.dumps({
    "auto_returned": len(result["written"]),
    "already_open": len(result["already_open"]),
}, ensure_ascii=False), file=sys.stderr)
PY
    fi
fi
echo "Completed approved canonical dataset: ${output_dir}" >&2
