#!/bin/bash


# ============================================================
# USER CONFIG
# ============================================================

STATION="13"

YEARS=("2022")
# YEARS=("2022" "2023")

# Number of runs processed by each Slurm array task.
CHUNK="10"

# true  = process the full dataset and exclude burn-sample events
# false = process only burn-sample events
FULL="true"

JSON_DIR="/mnt/nrdstor/hep/martinliu/data/realData/burnData/JSON_lists"

TRACKING_ROOT="/home/hep/martinliu/research/project_LDA/tracking_runs"

SLURM_SCRIPT="/home/hep/martinliu/research/project_LDA/applyHitFilter_chunks.slurm"


# Nodes that have repeatedly produced hanging jobs.
#
# Set to an empty string to disable node exclusion:
#
# EXCLUDE_NODES=""
#
EXCLUDE_NODES="c1424,c2005,c1213,c1406,c1908,c1926,c2015,c2018,c2009,c1514,c1921,c2021"


# ============================================================
# Optional rerun list
# ============================================================
#
# Normal submission:
#
#   bash applyHitFilter_chunks.sh
#
# Rerun previously unprocessed runs:
#
#   bash applyHitFilter_chunks.sh \
#       /path/to/unprocessed_runs.txt
#
# Rerun-list format:
#
#   YEAR RUN
#
# Example:
#
#   2022 105
#   2022 108
#   2023 412
# ============================================================

SOURCE_RUN_LIST="${1:-}"


# ============================================================
# Determine data mode and paths
# ============================================================

case "${FULL}" in

    true)

        DATA_MODE="full"

        DIR_OUT="/mnt/nrdstor/hep/martinliu/data/realData/project_LDA/filteredData/station${STATION}"

        JOB_NAME="filter_full_s${STATION}"

        ;;

    false)

        DATA_MODE="burn"

        DIR_OUT="/mnt/nrdstor/hep/martinliu/data/realData/project_LDA/burnData/filteredData/station${STATION}"

        JOB_NAME="filter_burn_s${STATION}"

        ;;

    *)

        echo "ERROR: FULL must be either true or false."
        exit 1

        ;;

esac


# Dynamic Slurm log filenames.
SLURM_OUT="/work/hep/martinliu/slurm_out/${JOB_NAME}-%A_%a.out"

SLURM_ERR="/work/hep/martinliu/slurm_out/${JOB_NAME}-%A_%a.err"


# ============================================================
# Validate configuration
# ============================================================

if [[ ! "${STATION}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid station number: ${STATION}"
    exit 1
fi


if [[ ! "${CHUNK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CHUNK must be a positive integer."
    exit 1
fi


if [[ ! -d "${JSON_DIR}" ]]; then
    echo "ERROR: JSON directory does not exist:"
    echo "${JSON_DIR}"
    exit 1
fi


if [[ ! -f "${SLURM_SCRIPT}" ]]; then
    echo "ERROR: Slurm worker script does not exist:"
    echo "${SLURM_SCRIPT}"
    exit 1
fi


# ============================================================
# Create unique tracking directory
# ============================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")


if [[ -n "${SOURCE_RUN_LIST}" ]]; then

    TRACKING_DIR="${TRACKING_ROOT}/filter_${DATA_MODE}_s${STATION}_rerun_${TIMESTAMP}"

else

    TRACKING_DIR="${TRACKING_ROOT}/filter_${DATA_MODE}_s${STATION}_${TIMESTAMP}"

fi


mkdir -p "${TRACKING_DIR}/processed"
mkdir -p "${TRACKING_DIR}/failed"
mkdir -p "${TRACKING_DIR}/input_missing"
mkdir -p "${TRACKING_DIR}/output_missing"
mkdir -p "${TRACKING_DIR}/task_status"

MKDIR_STATUS=$?


if (( MKDIR_STATUS != 0 )); then

    echo "ERROR: Could not create tracking directory:"
    echo "${TRACKING_DIR}"

    exit 1

fi


RUN_LIST="${TRACKING_DIR}/expected_runs.txt"

RUN_LIST_TMP="${TRACKING_DIR}/expected_runs.txt.tmp"


: > "${RUN_LIST_TMP}"


# ============================================================
# Build expected-run list
# ============================================================
#
# Format:
#
#   YEAR RUN
#
# ============================================================

if [[ -n "${SOURCE_RUN_LIST}" ]]; then

    # ========================================================
    # Rerun mode
    # ========================================================

    echo "Creating rerun list from:"
    echo "${SOURCE_RUN_LIST}"


    if [[ ! -s "${SOURCE_RUN_LIST}" ]]; then

        echo "ERROR: Rerun list does not exist or is empty:"
        echo "${SOURCE_RUN_LIST}"

        exit 1

    fi


    while read -r YEAR RUN EXTRA; do

        if [[ -z "${YEAR}" && -z "${RUN}" ]]; then
            continue
        fi


        if [[ -z "${YEAR}" || -z "${RUN}" || -n "${EXTRA}" ]]; then

            echo "ERROR: Malformed rerun-list entry:"
            echo "${YEAR} ${RUN} ${EXTRA}"

            exit 1

        fi


        if [[ ! "${YEAR}" =~ ^[0-9]{4}$ ]]; then

            echo "ERROR: Invalid year:"
            echo "${YEAR}"

            exit 1

        fi


        if [[ ! "${RUN}" =~ ^[0-9]+$ ]]; then

            echo "ERROR: Invalid run:"
            echo "${RUN}"

            exit 1

        fi


        JSON="${JSON_DIR}/station${STATION}_${YEAR}_burn_sample_evt_num_dict.json"


        if [[ ! -s "${JSON}" ]]; then

            echo "ERROR: JSON file required for this rerun is missing:"
            echo "${JSON}"

            exit 1

        fi


        RUN=$((10#${RUN}))


        echo "${YEAR} ${RUN}" \
            >> "${RUN_LIST_TMP}"

    done < "${SOURCE_RUN_LIST}"


else

    # ========================================================
    # Normal submission
    # ========================================================

    for YEAR in "${YEARS[@]}"; do

        JSON="${JSON_DIR}/station${STATION}_${YEAR}_burn_sample_evt_num_dict.json"


        if [[ ! -s "${JSON}" ]]; then

            echo "ERROR: JSON file does not exist or is empty:"
            echo "${JSON}"

            exit 1

        fi


        echo "Reading runs from:"
        echo "${JSON}"


        python - "${JSON}" "${YEAR}" >> "${RUN_LIST_TMP}" <<'PYTHON'
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
year = sys.argv[2]

try:
    with json_path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
except (OSError, json.JSONDecodeError) as exc:
    print(
        f"ERROR: Could not read {json_path}: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

if not isinstance(data, dict):
    print(
        f"ERROR: Expected the top level of {json_path} "
        f"to be a JSON object.",
        file=sys.stderr,
    )
    sys.exit(1)

runs = []

for run_key in data:
    try:
        run = int(run_key)
    except (TypeError, ValueError):
        print(
            f"ERROR: Invalid run-number key "
            f"{run_key!r} in {json_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    runs.append(run)

for run in sorted(set(runs)):
    print(year, run)
PYTHON

        PYTHON_STATUS=$?


        if (( PYTHON_STATUS != 0 )); then

            echo "ERROR: Could not extract run numbers from:"
            echo "${JSON}"

            exit 1

        fi

    done

fi


# ============================================================
# Sort and deduplicate run list
# ============================================================

sort -k1,1n -k2,2n -u \
    "${RUN_LIST_TMP}" \
    > "${RUN_LIST}"

SORT_STATUS=$?


rm -f "${RUN_LIST_TMP}"


if (( SORT_STATUS != 0 )); then
    echo "ERROR: Could not create sorted run list."
    exit 1
fi


N_RUNS=$(wc -l < "${RUN_LIST}")

N_RUNS=$(printf '%s' "${N_RUNS}" | tr -d '[:space:]')


if [[ ! "${N_RUNS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Could not determine number of runs."
    exit 1
fi


if (( N_RUNS == 0 )); then
    echo "ERROR: No runs were found."
    exit 1
fi


N_TASKS=$(( (N_RUNS + CHUNK - 1) / CHUNK ))


# ============================================================
# Create task map
# ============================================================
#
# Format:
#
#   TASK_INDEX START_LINE END_LINE
#
# Useful for seeing which runs belonged to a hanging task.
# ============================================================

TASK_MAP="${TRACKING_DIR}/task_map.txt"

: > "${TASK_MAP}"


for (( TASK_INDEX=0; TASK_INDEX<N_TASKS; TASK_INDEX++ )); do

    START_LINE=$((TASK_INDEX * CHUNK + 1))

    END_LINE=$((START_LINE + CHUNK - 1))


    if (( END_LINE > N_RUNS )); then
        END_LINE="${N_RUNS}"
    fi


    echo "${TASK_INDEX} ${START_LINE} ${END_LINE}" \
        >> "${TASK_MAP}"

done


# ============================================================
# Submission summary
# ============================================================

echo
echo "============================================================"
echo "Hit-filter submission"
echo "Station:             ${STATION}"
echo "Mode:                ${DATA_MODE}"
echo "FULL:                ${FULL}"
echo "Years:               ${YEARS[*]}"
echo "Runs per task:       ${CHUNK}"
echo "Number of runs:      ${N_RUNS}"
echo "Number of tasks:     ${N_TASKS}"
echo "Array range:         0-$((N_TASKS - 1))"
echo "Job name:            ${JOB_NAME}"
echo "Slurm output:        ${SLURM_OUT}"
echo "Slurm error:         ${SLURM_ERR}"
echo "Output directory:    ${DIR_OUT}"
echo "Tracking directory:  ${TRACKING_DIR}"
echo "Run list:            ${RUN_LIST}"
echo "Task map:            ${TASK_MAP}"

if [[ -n "${EXCLUDE_NODES}" ]]; then
    echo "Excluded nodes:      ${EXCLUDE_NODES}"
else
    echo "Excluded nodes:      none"
fi

echo "============================================================"


# ============================================================
# Save submission configuration
# ============================================================

cat > "${TRACKING_DIR}/submission_config.txt" << EOF
STATION=${STATION}
FULL=${FULL}
DATA_MODE=${DATA_MODE}
YEARS=${YEARS[*]}
CHUNK=${CHUNK}
N_RUNS=${N_RUNS}
N_TASKS=${N_TASKS}
DIR_OUT=${DIR_OUT}
JSON_DIR=${JSON_DIR}
RUN_LIST=${RUN_LIST}
TASK_MAP=${TASK_MAP}
TRACKING_DIR=${TRACKING_DIR}
SLURM_SCRIPT=${SLURM_SCRIPT}
SOURCE_RUN_LIST=${SOURCE_RUN_LIST}
EXCLUDE_NODES=${EXCLUDE_NODES}
JOB_NAME=${JOB_NAME}
SLURM_OUT=${SLURM_OUT}
SLURM_ERR=${SLURM_ERR}
SUBMISSION_TIME=$(date --iso-8601=seconds)
EOF


# ============================================================
# Create finalizer
# ============================================================
#
# processed_runs.txt:
#
#   YEAR RUN
#
# unprocessed_runs.txt:
#
#   YEAR RUN
#
# ============================================================

FINALIZE_SCRIPT="${TRACKING_DIR}/update_unprocessed_runs.sh"


cat > "${FINALIZE_SCRIPT}" << EOF
#!/bin/bash

EXPECTED_RUNS="${RUN_LIST}"

PROCESSED_DIR="${TRACKING_DIR}/processed"

PROCESSED_RUNS="${TRACKING_DIR}/processed_runs.txt"

UNPROCESSED_RUNS="${TRACKING_DIR}/unprocessed_runs.txt"


find "\${PROCESSED_DIR}" \
    -maxdepth 1 \
    -type f \
    -name "task_*.txt" \
    -exec cat {} + 2>/dev/null \
    | awk '
        NF == 2 &&
        \$1 ~ /^[0-9]{4}\$/ &&
        \$2 ~ /^[0-9]+\$/ {

            key = \$1 FS \$2

            if (!(key in seen)) {
                print \$1, \$2
                seen[key] = 1
            }
        }
    ' \
    | sort -k1,1n -k2,2n \
    > "\${PROCESSED_RUNS}"


awk '
    NR == FNR {
        processed[\$1 FS \$2] = 1
        next
    }

    !((\$1 FS \$2) in processed) {
        print \$1, \$2
    }
' "\${PROCESSED_RUNS}" \
  "\${EXPECTED_RUNS}" \
  | sort -k1,1n -k2,2n \
  > "\${UNPROCESSED_RUNS}"


EXPECTED_COUNT=\$(wc -l < "\${EXPECTED_RUNS}")

PROCESSED_COUNT=\$(wc -l < "\${PROCESSED_RUNS}")

UNPROCESSED_COUNT=\$(wc -l < "\${UNPROCESSED_RUNS}")


echo
echo "============================================================"
echo "Expected runs:       \${EXPECTED_COUNT}"
echo "Processed runs:      \${PROCESSED_COUNT}"
echo "Unprocessed runs:    \${UNPROCESSED_COUNT}"
echo "Processed list:      \${PROCESSED_RUNS}"
echo "Unprocessed list:    \${UNPROCESSED_RUNS}"
echo "============================================================"


if (( UNPROCESSED_COUNT > 0 )); then

    echo
    echo "Runs still requiring processing:"

    cat "\${UNPROCESSED_RUNS}"

fi
EOF


chmod +x "${FINALIZE_SCRIPT}"


# ============================================================
# Submit Slurm array
# ============================================================

echo
echo "Submitting Slurm array..."


if [[ -n "${EXCLUDE_NODES}" ]]; then

    SBATCH_RESULT=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --output="${SLURM_OUT}" \
        --error="${SLURM_ERR}" \
        --exclude="${EXCLUDE_NODES}" \
        --array="0-$((N_TASKS - 1))" \
        --export="ALL,STATION=${STATION},FULL=${FULL},DATA_MODE=${DATA_MODE},CHUNK=${CHUNK},DIR_OUT=${DIR_OUT},JSON_DIR=${JSON_DIR},RUN_LIST=${RUN_LIST},TRACKING_DIR=${TRACKING_DIR}" \
        "${SLURM_SCRIPT}")

else

    SBATCH_RESULT=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --output="${SLURM_OUT}" \
        --error="${SLURM_ERR}" \
        --array="0-$((N_TASKS - 1))" \
        --export="ALL,STATION=${STATION},FULL=${FULL},DATA_MODE=${DATA_MODE},CHUNK=${CHUNK},DIR_OUT=${DIR_OUT},JSON_DIR=${JSON_DIR},RUN_LIST=${RUN_LIST},TRACKING_DIR=${TRACKING_DIR}" \
        "${SLURM_SCRIPT}")

fi


SBATCH_STATUS=$?


if (( SBATCH_STATUS != 0 )); then

    echo "ERROR: Slurm submission failed."

    exit 1

fi


ARRAY_JOB_ID="${SBATCH_RESULT%%;*}"


if [[ ! "${ARRAY_JOB_ID}" =~ ^[0-9]+$ ]]; then

    echo "ERROR: Invalid Slurm job ID returned:"
    echo "${SBATCH_RESULT}"

    exit 1

fi


echo "${ARRAY_JOB_ID}" \
    > "${TRACKING_DIR}/slurm_job_id.txt"


# ============================================================
# Submit automatic finalizer
# ============================================================
#
# Runs after every array task has completed, failed, or been
# canceled.
# ============================================================

FINALIZER_RESULT=$(sbatch --parsable \
    --dependency="afterany:${ARRAY_JOB_ID}" \
    --job-name="finalize_${DATA_MODE}_s${STATION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=500M \
    --time=00:15:00 \
    --output="${TRACKING_DIR}/finalize-%j.out" \
    --error="${TRACKING_DIR}/finalize-%j.err" \
    --wrap="bash ${FINALIZE_SCRIPT}")


FINALIZER_STATUS=$?


if (( FINALIZER_STATUS == 0 )); then

    FINALIZER_JOB_ID="${FINALIZER_RESULT%%;*}"


    echo "${FINALIZER_JOB_ID}" \
        > "${TRACKING_DIR}/finalizer_job_id.txt"

else

    echo "WARNING: Could not submit automatic finalizer."

    echo "Run manually after the array terminates:"

    echo
    echo "bash ${FINALIZE_SCRIPT}"

fi


# ============================================================
# Final summary
# ============================================================

echo
echo "============================================================"
echo "Hit-filter array submitted successfully."
echo "Array job ID:       ${ARRAY_JOB_ID}"
echo "Station:            ${STATION}"
echo "Mode:               ${DATA_MODE}"
echo "Number of runs:     ${N_RUNS}"
echo "Runs per task:      ${CHUNK}"
echo "Number of tasks:    ${N_TASKS}"
echo "Job name:           ${JOB_NAME}"

if [[ -n "${EXCLUDE_NODES}" ]]; then
    echo "Excluded nodes:     ${EXCLUDE_NODES}"
fi

echo "Tracking directory: ${TRACKING_DIR}"
echo
echo "After the array ends:"
echo "  ${TRACKING_DIR}/processed_runs.txt"
echo "  ${TRACKING_DIR}/unprocessed_runs.txt"
echo
echo "Manual update command:"
echo "  bash ${FINALIZE_SCRIPT}"
echo "============================================================"
