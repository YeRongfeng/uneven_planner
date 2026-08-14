#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 GRADE MAP_LIST.txt OUTPUT_DIR ROS_MASTER_PORT" >&2
    exit 2
fi

grade="$1"
map_list="$(realpath "$2")"
mkdir -p "$3"
output_dir="$(realpath "$3")"
master_port="$4"
paths_per_map="${PATHS_PER_MAP:-3}"
attempts_per_path="${ATTEMPTS_PER_PATH:-5}"
seed_base="${SEED_BASE:-2026081600}"
poll_seconds="${POLL_SECONDS:-2}"
max_polls="${MAX_MAP_POLLS:-300}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "${script_dir}/../../../.." && pwd)"
if [[ ! -f "${map_list}" ]]; then
    echo "Missing map list: ${map_list}" >&2
    exit 1
fi
mapfile -t maps < <(sed '/^[[:space:]]*$/d' "${map_list}")
if [[ ${#maps[@]} -eq 0 ]]; then
    echo "Map list is empty: ${map_list}" >&2
    exit 1
fi

source "${workspace}/devel/setup.bash"
export ROS_MASTER_URI="http://localhost:${master_port}"

stop_launch() {
    if [[ -n "${launch_pid:-}" ]] && kill -0 "${launch_pid}" 2>/dev/null; then
        kill -INT -- "-${launch_pid}" 2>/dev/null || true
        wait "${launch_pid}" 2>/dev/null || true
    fi
    launch_pid=""
}
trap stop_launch EXIT INT TERM

for index in "${!maps[@]}"; do
    map_path="$(realpath "${maps[$index]}")"
    scene_name="scene_$(printf '%03d' "${index}")"
    scene_output="${output_dir}/${scene_name}"
    if [[ -e "${scene_output}" ]]; then
        echo "Refusing to overwrite calibration output: ${scene_output}" >&2
        exit 1
    fi
    mkdir -p "${scene_output}/ros_logs" "${scene_output}/matplotlib" \
        "${scene_output}/dataset"
    export ROS_LOG_DIR="${scene_output}/ros_logs"
    export MPLCONFIGDIR="${scene_output}/matplotlib"

    echo "[${grade} ${index}/${#maps[@]}] ${map_path}"
    setsid roslaunch plan_manager terrain_dataset_generation_parallel.launch \
        parallel_workers:=1 \
        num_environments:=1 \
        train_paths_per_env:="${paths_per_map}" \
        val_paths_per_env:=1 \
        stop_after_train:=true \
        dataset_dir:="${scene_output}/dataset" \
        start_env_id:=0 \
        external_map_path:="${map_path}" \
        external_map_paths:="${map_path}" \
        train_external_map_paths:="${map_path}" \
        val_external_map_paths:="${map_path}" \
        external_map_is_canonical:=true \
        canonical_maps_per_environment:=1 \
        canonical_primary_scene_count:=1 \
        canonical_pool_start_env_id:=0 \
        target_map_size:=20.0 \
        target_resolution:=0.2 \
        generation_random_seed:="$((seed_base + index))" \
        max_path_retries_before_regenerate:="${attempts_per_path}" \
        >"${scene_output}/roslaunch.log" 2>&1 &
    launch_pid=$!

    finished=false
    for ((poll=1; poll<=max_polls; poll++)); do
        manifest="${scene_output}/dataset/experiment_manifest.json"
        if [[ -f "${manifest}" ]]; then
            status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${manifest}")"
            if [[ "${status}" == "completed" || "${status}" == "map_rejected" ]]; then
                finished=true
                break
            fi
        fi
        if ! kill -0 "${launch_pid}" 2>/dev/null; then
            break
        fi
        sleep "${poll_seconds}"
    done
    stop_launch
    if [[ "${finished}" != true ]]; then
        echo "Calibration did not finish for ${map_path}; inspect ${scene_output}/roslaunch.log" >&2
        exit 1
    fi
done

trap - EXIT INT TERM
echo "Completed Stage-C grade calibration: ${grade}, maps=${#maps[@]}, paths_per_map=${paths_per_map}"
