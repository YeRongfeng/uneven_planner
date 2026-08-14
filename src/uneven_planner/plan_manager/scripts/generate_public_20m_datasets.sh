#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "${script_dir}/../../../.." && pwd)"
output_root="${1:-${workspace}/dataset/public_terrain_20m_v1}"
if [[ $# -gt 0 ]]; then
    shift
fi
domains=("$@")
if [[ ${#domains[@]} -eq 0 ]]; then
    domains=(desert forest hill snow vocano)
fi

scenes="${SCENES_PER_SPLIT:-100}"
train_paths="${TRAIN_PATHS_PER_SCENE:-100}"
val_paths="${VAL_PATHS_PER_SCENE:-10}"
parallel_workers="${PARALLEL_WORKERS:-8}"
canonical_candidates="${CANONICAL_CANDIDATES_PER_ENV:-2}"
max_attempts="${MAX_ATTEMPTS_PER_SPLIT:-$((scenes * canonical_candidates * 30))}"
dry_run="${DRY_RUN:-0}"

run_domain() {
    local requested_domain="$1"
    local domain output_name train_laz val_laz train_site val_site
    local train_url val_url train_crs val_crs seed ros_port
    case "${requested_domain}" in
        desert)
            domain="desert"; output_name="desert"; seed=20260901; ros_port=11401
            train_laz="${workspace}/dataset/external/white_sands_nm10/raw/ot_381000_3632000_smrf.laz"
            val_laz="${workspace}/dataset/external/val_desert_ca22/raw/710000_3755000.laz"
            train_site="NM10_Kocurek"; val_site="CA22_Marvin"
            train_url="https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.022012.26913.1"
            val_url="https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.062023.6340.1"
            train_crs="EPSG:26913+5703"; val_crs="EPSG:6340"
            ;;
        forest)
            domain="forest"; output_name="forest"; seed=20260902; ros_port=11402
            train_laz="${workspace}/dataset/external/forest_wa21/raw/642000_5265000.laz"
            val_laz="${workspace}/dataset/external/forest_az17_donager/raw/C429000_3888000.laz"
            train_site="WA21_Lumbrazo"; val_site="AZ17_Donager"
            train_url="https://portal.opentopography.org/lidarDataset?opentopoID=OTLAS.122021.6339.1"
            val_url="https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.062018.26912.1"
            train_crs="EPSG:6339"; val_crs="EPSG:26912"
            ;;
        hill)
            domain="hill"; output_name="hill"; seed=20260903; ros_port=11403
            train_laz="${workspace}/dataset/external/hill_co10/raw/ot_480000_4354000.laz"
            val_laz="${workspace}/dataset/external/hill_ca13_johnstone/raw/ot_677000_4007000_1.laz"
            train_site="CO10_Tucker"; val_site="CA13_Johnstone"
            train_url="https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.052013.26913.2"
            val_url="https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.052014.26910.2"
            train_crs="EPSG:26913+5703"; val_crs="EPSG:26910+5703"
            ;;
        snow)
            domain="snow"; output_name="snow"; seed=20260904; ros_port=11404
            train_laz="${workspace}/dataset/external/snow_bczo/raw/ot_440000_4420000.laz"
            val_laz="${workspace}/dataset/external/snow_ca22_piske/raw/731000_4366000.laz"
            train_site="BCZO_SnowOn09"; val_site="CA22_Piske"
            train_url="https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.082016.26913.1"
            val_url="https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.112022.6339.1"
            train_crs="EPSG:26913+5703"; val_crs="EPSG:6339"
            ;;
        volcano|vocano)
            domain="volcano"; output_name="vocano"; seed=20260905; ros_port=11405
            train_laz="${workspace}/dataset/external/volcano_ca16/raw/ot_C321000_4174000.laz"
            val_laz="${workspace}/dataset/external/val_volcano_ca19/raw/624000_4488000.laz"
            train_site="CA16_Martens"; val_site="CA19_Herbst"
            train_url="https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.032017.26911.1"
            val_url="https://portal.opentopography.org/lidarDataset?opentopoID=OTLAS.072020.6339.1"
            train_crs="EPSG:26911+5703"; val_crs="EPSG:6339"
            ;;
        *)
            echo "Unknown domain: ${requested_domain}" >&2
            return 2
            ;;
    esac

    for source_path in "${train_laz}" "${val_laz}"; do
        if [[ ! -f "${source_path}" ]]; then
            echo "Missing source LAZ: ${source_path}" >&2
            return 1
        fi
    done
    echo "${output_name}: ${train_site} -> ${val_site}; scenes=${scenes}+${scenes}, paths=${train_paths}/${val_paths}, candidates=${canonical_candidates}, max_sampling_attempts=${max_attempts}"
    if [[ "${dry_run}" == "1" ]]; then
        return
    fi

    env \
        ROS_MASTER_URI="http://localhost:${ros_port}" \
        DOMAIN="${domain}" \
        TRAIN_SITE_ID="${train_site}" VAL_SITE_ID="${val_site}" \
        TRAIN_SOURCE_PROFILE=als VAL_SOURCE_PROFILE=als \
        TRAIN_POINT_CLASSES=2 VAL_POINT_CLASSES=2 \
        TRAIN_SOURCE_URL="${train_url}" VAL_SOURCE_URL="${val_url}" \
        TRAIN_LICENSE="CC BY 4.0" VAL_LICENSE="CC BY 4.0" \
        TRAIN_CRS="${train_crs}" VAL_CRS="${val_crs}" \
        SEED="${seed}" PARALLEL_WORKERS="${parallel_workers}" \
        CANONICAL_CANDIDATES_PER_ENV="${canonical_candidates}" \
        MAX_ATTEMPTS_PER_SPLIT="${max_attempts}" \
        "${script_dir}/generate_from_laz_mother_maps.sh" \
        "${train_laz}" "${val_laz}" "${output_root}/${output_name}" \
        "${scenes}" "${train_paths}" "${val_paths}"

    python3 "${script_dir}/verify_public_20m_dataset.py" \
        "${output_root}/${output_name}" \
        --scenes "${scenes}" \
        --train-paths "${train_paths}" \
        --val-paths "${val_paths}" \
        --train-site "${train_site}" \
        --val-site "${val_site}"
}

mkdir -p "${output_root}"
for domain in "${domains[@]}"; do
    run_domain "${domain}"
done
