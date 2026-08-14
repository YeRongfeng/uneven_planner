#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
maps_dir="${workspace}/src/uneven_planner/uneven_map/maps"
output_root="${1:-${workspace}/src/uneven_planner/plan_manager/datasets/terrain_20m_100scenes}"
log_root="${output_root}/logs"

terrains=(desert forest hill snow vocano)
seeds=(13001 23001 33001 43001 53001)

train_scenes=100
val_scenes=100
train_paths_per_scene=100
val_paths_per_scene=10
parallel_workers=8
poll_seconds=15
max_polls=2880

mkdir -p "${output_root}" "${log_root}"
source "${workspace}/devel/setup.bash"

active_launch_pid=""
stop_active_launch() {
    if [[ -n "${active_launch_pid}" ]] && kill -0 "${active_launch_pid}" 2>/dev/null; then
        kill -INT -- "-${active_launch_pid}" 2>/dev/null || true
        wait "${active_launch_pid}" 2>/dev/null || true
    fi
    active_launch_pid=""
}
trap stop_active_launch EXIT INT TERM

count_files() {
    local directory="$1"
    local pattern="$2"
    if [[ ! -d "${directory}" ]]; then
        printf '0\n'
        return
    fi
    find "${directory}" -type f -name "${pattern}" -printf '.' | wc -c
}

for index in "${!terrains[@]}"; do
    terrain="${terrains[$index]}"
    seed="${seeds[$index]}"
    source_map="${maps_dir}/${terrain}.pcd"
    final_dir="${output_root}/${terrain}"
    partial_dir="${output_root}/${terrain}.partial"
    terrain_log_dir="${log_root}/${terrain}"
    run_log="${terrain_log_dir}/generation.log"

    if [[ ! -f "${source_map}" ]]; then
        echo "Missing source map: ${source_map}" >&2
        exit 1
    fi
    if [[ -e "${final_dir}" ]]; then
        echo "Refusing to overwrite completed output: ${final_dir}" >&2
        exit 1
    fi
    if [[ -e "${partial_dir}" ]]; then
        echo "Refusing to overwrite partial output: ${partial_dir}" >&2
        exit 1
    fi

    mkdir -p "${partial_dir}" "${terrain_log_dir}/ros" "${terrain_log_dir}/matplotlib"
    echo "[$(date '+%F %T')] Starting ${terrain}: train=100x100 paths, val=100x10 paths"

    setsid env \
        ROS_MASTER_URI="http://localhost:11321" \
        ROS_LOG_DIR="${terrain_log_dir}/ros" \
        MPLCONFIGDIR="${terrain_log_dir}/matplotlib" \
        roslaunch plan_manager terrain_dataset_generation_parallel.launch \
        parallel_workers:="${parallel_workers}" \
        num_environments:="${train_scenes}" \
        train_paths_per_env:="${train_paths_per_scene}" \
        val_paths_per_env:="${val_paths_per_scene}" \
        dataset_dir:="${partial_dir}" \
        start_env_id:=0 \
        external_map_path:="${source_map}" \
        external_map_paths:="${source_map}" \
        external_map_min_physical_size:=26.0 \
        target_map_size:=20.0 \
        target_resolution:=0.2 \
        crop_random_seed:="${seed}" \
        enable_rviz:=false >"${run_log}" 2>&1 &
    active_launch_pid=$!

    for ((poll=1; poll<=max_polls; poll++)); do
        train_maps="$(count_files "${partial_dir}/train" 'map.p')"
        val_maps="$(count_files "${partial_dir}/val" 'map.p')"
        train_paths="$(count_files "${partial_dir}/train" 'path_*.p')"
        val_paths="$(count_files "${partial_dir}/val" 'path_*.p')"

        if ((poll == 1 || poll % 4 == 0)); then
            echo "[$(date '+%F %T')] ${terrain}: maps ${train_maps}/100 train, ${val_maps}/100 val; paths ${train_paths}/10000 train, ${val_paths}/1000 val"
        fi

        if [[ "${train_maps}" -eq "${train_scenes}" &&
              "${val_maps}" -eq "${val_scenes}" &&
              "${train_paths}" -eq 10000 &&
              "${val_paths}" -eq 1000 ]]; then
            sleep 5
            stop_active_launch
            mv "${partial_dir}" "${final_dir}"
            echo "[$(date '+%F %T')] Completed ${terrain}: ${final_dir}"
            break
        fi

        if grep -q "process has died" "${run_log}"; then
            echo "ROS worker failed for ${terrain}; inspect ${run_log}" >&2
            exit 1
        fi
        if ! kill -0 "${active_launch_pid}" 2>/dev/null; then
            echo "roslaunch exited before ${terrain} completed; inspect ${run_log}" >&2
            exit 1
        fi
        if [[ "${poll}" -eq "${max_polls}" ]]; then
            echo "Timed out while generating ${terrain}; partial data kept at ${partial_dir}" >&2
            exit 1
        fi
        sleep "${poll_seconds}"
    done
done

trap - EXIT INT TERM
echo "[$(date '+%F %T')] All requested datasets completed under ${output_root}"
