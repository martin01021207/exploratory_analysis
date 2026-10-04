#!/bin/bash


# ============================================================
# User configuration
# ============================================================

STATION="13"

# Number of simulation runs processed by each array task.
#
# CHUNK="1" means one energy/run pair per array task.
# CHUNK="10" means up to ten energy/run pairs per array task.
CHUNK="20"

ENERGIES=(
    16.0
    16.5
    17.0
    17.5
    18.0
    18.5
    19.0
)

BASE_DIR="/mnt/nrdstor/hep/martinliu/data/simCR/proxy/reduced_sets/station${STATION}"

DIR_OUT_BASE="/mnt/nrdstor/hep/martinliu/data/simData/CR/project_LDA/simPreprocessed/filteredData/station${STATION}"

TRACKING_ROOT="/home/hep/martinliu/research/project_LDA/tracking_runs"

SLURM_SCRIPT="/home/hep/martinliu/research/project_LDA/applyHitFilter_CR_proxy.slurm"


# Nodes that have repeatedly produced hanging jobs.
#
# Set this to an empty string to disable node exclusion:
#
# EXCLUDE_NODES=""
#
EXCLUDE_NODES="c1424,c2005,c1213,c1406,c1908,c1926,c2015,c2018,c2009,c1514,c1921,c2021"


# Dynamic Slurm naming.
JOB_NAME="filter_proxy_s${STATION}"

SLURM_OUT="/work/hep/martinliu/slurm_out/${JOB_NAME}-%A_%a.out"

SLURM_ERR="/work/hep/martinliu/slurm_out/${JOB_NAME}-%A_%a.err"


# ============================================================
# Optional rerun list
# ============================================================
#
# Normal submission:
#
#     bash applyHitFilter_CR_proxy.sh
#
# Rerun only previously unprocessed runs:
#
#     bash applyHitFilter_CR_proxy.sh \
#         /path/to/unprocessed_runs.txt
#
# The rerun file must contain:
#
#     ENERGY RUN
#
# Example:
#
#     16.0 84
#     16.0 555
#     17.5 21
# ============================================================

SOURCE_RUN_LIST="${1:-}"


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


if [[ ! -d "${BASE_DIR}" ]]; then
    echo "ERROR: Base input directory does not exist:"
    echo "${BASE_DIR}"
    exit 1
fi


if [[ ! -f "${SLURM_SCRIPT}" ]]; then
    echo "ERROR: Slurm script does not exist:"
    echo "${SLURM_SCRIPT}"
    exit 1
fi


# ============================================================
# Create a unique tracking directory
# ============================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")


if [[ -n "${SOURCE_RUN_LIST}" ]]; then

    TRACKING_DIR="${TRACKING_ROOT}/filter_proxy_s${STATION}_rerun_${TIMESTAMP}"

else

    TRACKING_DIR="${TRACKING_ROOT}/filter_proxy_s${STATION}_${TIMESTAMP}"

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
# Create expected-run list
# ============================================================
#
# Format:
#
#     ENERGY RUN
#
# ============================================================

if [[ -n "${SOURCE_RUN_LIST}" ]]; then

    # ========================================================
    # Rerun mode
    # ========================================================

    echo "Creating rerun list from:"
    echo "${SOURCE_RUN_LIST}"


    if [[ ! -s "${SOURCE_RUN_LIST}" ]]; then
        echo "ERROR: Supplied rerun list does not exist or is empty:"
        echo "${SOURCE_RUN_LIST}"
        exit 1
    fi


    awk '
        NF == 0 {
            next
        }

        NF != 2 {
            print "ERROR: Malformed line:", $0 > "/dev/stderr"
            bad = 1
            next
        }

        $1 !~ /^[0-9]+([.][0-9]+)?$/ {
            print "ERROR: Invalid energy:", $0 > "/dev/stderr"
            bad = 1
            next
        }

        $2 !~ /^[0-9]+$/ {
            print "ERROR: Invalid run number:", $0 > "/dev/stderr"
            bad = 1
            next
        }

        {
            print $1, $2
        }

        END {
            if (bad) {
                exit 1
            }
        }
    ' "${SOURCE_RUN_LIST}" > "${RUN_LIST_TMP}"


    VALIDATION_STATUS=$?


    if (( VALIDATION_STATUS != 0 )); then
        echo "ERROR: The supplied rerun list contains invalid entries."
        exit 1
    fi


else

    # ========================================================
    # Normal full scan
    # ========================================================

    echo "Scanning proxy simulation input files."


    for ENERGY in "${ENERGIES[@]}"; do

        DIR_IN="${BASE_DIR}/lgE${ENERGY}"


        if [[ ! -d "${DIR_IN}" ]]; then
            echo "WARNING: Input directory does not exist:"
            echo "${DIR_IN}"
            continue
        fi


        mapfile -t RUNS < <(
            find "${DIR_IN}" \
                -maxdepth 1 \
                -type f \
                -size +0c \
                -name "*.nur" \
                | sed -nE 's/.*_[jc]([0-9]+).*\.nur$/\1/p' \
                | sort -n \
                | uniq
        )


        N_RUNS_ENERGY=${#RUNS[@]}

        echo "ENERGY=${ENERGY}, N_RUNS=${N_RUNS_ENERGY}"


        for RUN_STR in "${RUNS[@]}"; do

            if [[ ! "${RUN_STR}" =~ ^[0-9]+$ ]]; then
                echo "WARNING: Invalid run extracted from filename:"
                echo "${RUN_STR}"
                continue
            fi


            RUN=$((10#${RUN_STR}))


            echo "${ENERGY} ${RUN}" \
                >> "${RUN_LIST_TMP}"

        done

    done

fi


# ============================================================
# Sort and deduplicate energy/run pairs
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
    echo "ERROR: No simulation runs were found."
    exit 1
fi


N_TASKS=$(( (N_RUNS + CHUNK - 1) / CHUNK ))


# ============================================================
# Create task map
# ============================================================
#
# Format:
#
#     TASK_INDEX START_LINE END_LINE
#
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
echo "Proxy hit-filter submission"
echo "Station:             ${STATION}"
echo "Total energy/runs:   ${N_RUNS}"
echo "Runs per task:       ${CHUNK}"
echo "Number of tasks:     ${N_TASKS}"
echo "Array range:         0-$((N_TASKS - 1))"
echo "Job name:            ${JOB_NAME}"
echo "Input base:          ${BASE_DIR}"
echo "Output base:         ${DIR_OUT_BASE}"
echo "Tracking directory:  ${TRACKING_DIR}"
echo "Expected-run list:   ${RUN_LIST}"
echo "Task map:            ${TASK_MAP}"

if [[ -n "${EXCLUDE_NODES}" ]]; then
    echo "Excluded nodes:      ${EXCLUDE_NODES}"
else
    echo "Excluded nodes:      none"
fi

echo "============================================================"


echo
echo "Runs per energy:"


awk '
    {
        count[$1]++
    }

    END {
        for (energy in count) {
            print energy, count[energy]
        }
    }
' "${RUN_LIST}" | sort -k1,1n


echo
echo "First ten entries:"

head -n 10 "${RUN_LIST}"


# ============================================================
# Save submission configuration
# ============================================================

cat > "${TRACKING_DIR}/submission_config.txt" << EOF
STATION=${STATION}
ENERGIES=${ENERGIES[*]}
CHUNK=${CHUNK}
N_RUNS=${N_RUNS}
N_TASKS=${N_TASKS}
BASE_DIR=${BASE_DIR}
DIR_OUT_BASE=${DIR_OUT_BASE}
EXCLUDE_NODES=${EXCLUDE_NODES}
RUN_LIST=${RUN_LIST}
TASK_MAP=${TASK_MAP}
TRACKING_DIR=${TRACKING_DIR}
SLURM_SCRIPT=${SLURM_SCRIPT}
SOURCE_RUN_LIST=${SOURCE_RUN_LIST}
JOB_NAME=${JOB_NAME}
SLURM_OUT=${SLURM_OUT}
SLURM_ERR=${SLURM_ERR}
SUBMISSION_TIME=$(date --iso-8601=seconds)
EOF


# ============================================================
# Create finalization script
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
        \$1 ~ /^[0-9]+([.][0-9]+)?\$/ &&
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
echo "Expected energy/run pairs: \${EXPECTED_COUNT}"
echo "Processed:                 \${PROCESSED_COUNT}"
echo "Unprocessed:               \${UNPROCESSED_COUNT}"
echo "Processed list:            \${PROCESSED_RUNS}"
echo "Unprocessed list:          \${UNPROCESSED_RUNS}"
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
echo "Submitting array:"
echo "0-$((N_TASKS - 1))"


if [[ -n "${EXCLUDE_NODES}" ]]; then

    SBATCH_RESULT=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --output="${SLURM_OUT}" \
        --error="${SLURM_ERR}" \
        --exclude="${EXCLUDE_NODES}" \
        --array="0-$((N_TASKS - 1))" \
        --export="ALL,STATION=${STATION},CHUNK=${CHUNK},BASE_DIR=${BASE_DIR},DIR_OUT_BASE=${DIR_OUT_BASE},TASK_LIST=${RUN_LIST},TRACKING_DIR=${TRACKING_DIR}" \
        "${SLURM_SCRIPT}")

else

    SBATCH_RESULT=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --output="${SLURM_OUT}" \
        --error="${SLURM_ERR}" \
        --array="0-$((N_TASKS - 1))" \
        --export="ALL,STATION=${STATION},CHUNK=${CHUNK},BASE_DIR=${BASE_DIR},DIR_OUT_BASE=${DIR_OUT_BASE},TASK_LIST=${RUN_LIST},TRACKING_DIR=${TRACKING_DIR}" \
        "${SLURM_SCRIPT}")

fi


SBATCH_STATUS=$?


if (( SBATCH_STATUS != 0 )); then
    echo "ERROR: Slurm array submission failed."
    exit 1
fi


ARRAY_JOB_ID="${SBATCH_RESULT%%;*}"


if [[ ! "${ARRAY_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid Slurm array job ID returned:"
    echo "${SBATCH_RESULT}"
    exit 1
fi


echo "${ARRAY_JOB_ID}" \
    > "${TRACKING_DIR}/slurm_job_id.txt"


# ============================================================
# Submit automatic finalizer
# ============================================================

FINALIZER_RESULT=$(sbatch --parsable \
    --dependency="afterany:${ARRAY_JOB_ID}" \
    --job-name="finalize_proxy_s${STATION}" \
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
    echo "Run this manually after the array ends:"
    echo
    echo "bash ${FINALIZE_SCRIPT}"

fi


# ============================================================
# Final summary
# ============================================================

echo
echo "============================================================"
echo "Proxy array submitted successfully."
echo "Array job ID:       ${ARRAY_JOB_ID}"
echo "Station:            ${STATION}"
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
