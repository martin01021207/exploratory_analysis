#!/bin/bash


# ============================================================
# User configuration
# ============================================================

STATION="13"

# true  = full data
# false = burn-sample data
FULL="true"

# Number of runs processed by each array task.
CHUNK="5"

TRACKING_ROOT="/home/hep/martinliu/research/project_LDA/tracking_runs"

SLURM_SCRIPT="/home/hep/martinliu/research/project_LDA/makeVariables_chunks.slurm"


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
#   bash makeVariables_chunks.sh
#
# Rerun only previously unprocessed runs:
#
#   bash makeVariables_chunks.sh \
#       /path/to/unprocessed_runs.txt
#
# Rerun-list format:
#
#   RUN
#
# Example:
#
#   137
#   205
#   812
# ============================================================

SOURCE_RUN_LIST="${1:-}"


# ============================================================
# Select full-data or burn-sample paths
# ============================================================

if [[ "${FULL}" == "true" ]]; then

    DATA_MODE="full"

    IN_DIR="/mnt/nrdstor/hep/martinliu/data/realData/project_LDA/filteredData/station${STATION}"

    OUT_DIR="/mnt/nrdstor/hep/martinliu/data/realData/project_LDA/variableFiles/station${STATION}"

    JOB_NAME="vars_full_s${STATION}"

elif [[ "${FULL}" == "false" ]]; then

    DATA_MODE="burn"

    IN_DIR="/mnt/nrdstor/hep/martinliu/data/realData/project_LDA/burnData/filteredData/station${STATION}"

    OUT_DIR="/mnt/nrdstor/hep/martinliu/data/realData/project_LDA/burnData/variableFiles/station${STATION}"

    JOB_NAME="vars_burn_s${STATION}"

else

    echo "ERROR: FULL must be either true or false."
    exit 1

fi


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


if [[ ! -d "${IN_DIR}" ]]; then
    echo "ERROR: Input directory does not exist:"
    echo "${IN_DIR}"
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

    TRACKING_DIR="${TRACKING_ROOT}/makeVariables_${DATA_MODE}_s${STATION}_rerun_${TIMESTAMP}"

else

    TRACKING_DIR="${TRACKING_ROOT}/makeVariables_${DATA_MODE}_s${STATION}_${TIMESTAMP}"

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


RUNLIST="${TRACKING_DIR}/expected_runs.txt"

RUNLIST_TMP="${TRACKING_DIR}/expected_runs.txt.tmp"


: > "${RUNLIST_TMP}"


# ============================================================
# Build expected-run list
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


    while read -r RUN EXTRA; do

        if [[ -z "${RUN}" ]]; then
            continue
        fi


        if [[ -n "${EXTRA}" ]]; then

            echo "ERROR: Malformed rerun-list entry:"
            echo "${RUN} ${EXTRA}"

            exit 1

        fi


        if [[ ! "${RUN}" =~ ^[0-9]+$ ]]; then

            echo "ERROR: Invalid run number:"
            echo "${RUN}"

            exit 1

        fi


        RUN=$((10#${RUN}))


        INPUT_FILE="${IN_DIR}/filtered_s${STATION}_r${RUN}.root"


        if [[ ! -s "${INPUT_FILE}" ]]; then

            echo "ERROR: Filtered input file for rerun is missing or empty:"
            echo "${INPUT_FILE}"

            exit 1

        fi


        echo "${RUN}" \
            >> "${RUNLIST_TMP}"

    done < "${SOURCE_RUN_LIST}"


else

    # ========================================================
    # Normal submission
    # ========================================================

    echo "Creating run list from:"
    echo "${IN_DIR}"


    find "${IN_DIR}" \
        -maxdepth 1 \
        -type f \
        -size +0c \
        -name "filtered_s${STATION}_r*.root" \
        -printf '%f\n' \
        | sed -nE "s/^filtered_s${STATION}_r0*([0-9]+)\.root$/\1/p" \
        | awk '{print $1 + 0}' \
        | sort -n \
        | uniq \
        > "${RUNLIST_TMP}"

    FIND_STATUS=$?


    if (( FIND_STATUS != 0 )); then

        echo "ERROR: Failed while creating the run list."

        rm -f "${RUNLIST_TMP}"

        exit 1

    fi

fi


# ============================================================
# Sort and deduplicate run list
# ============================================================

sort -n -u \
    "${RUNLIST_TMP}" \
    > "${RUNLIST}"

SORT_STATUS=$?


rm -f "${RUNLIST_TMP}"


if (( SORT_STATUS != 0 )); then
    echo "ERROR: Could not create the final run list."
    exit 1
fi


NRUNS=$(wc -l < "${RUNLIST}")

NRUNS=$(printf '%s' "${NRUNS}" | tr -d '[:space:]')


if [[ ! "${NRUNS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Could not determine number of runs."
    exit 1
fi


if (( NRUNS == 0 )); then
    echo "ERROR: No runs were found."
    exit 1
fi


NTASKS=$(( (NRUNS + CHUNK - 1) / CHUNK ))


# ============================================================
# Create task map
# ============================================================
#
# Format:
#
#   TASK_INDEX START_LINE END_LINE
#
# This makes it easy to identify which runs were assigned to a
# hanging task.
# ============================================================

TASK_MAP="${TRACKING_DIR}/task_map.txt"

: > "${TASK_MAP}"


for (( TASK_INDEX=0; TASK_INDEX<NTASKS; TASK_INDEX++ )); do

    START_LINE=$((TASK_INDEX * CHUNK + 1))

    END_LINE=$((START_LINE + CHUNK - 1))


    if (( END_LINE > NRUNS )); then
        END_LINE="${NRUNS}"
    fi


    echo "${TASK_INDEX} ${START_LINE} ${END_LINE}" \
        >> "${TASK_MAP}"

done


# ============================================================
# Print submission summary
# ============================================================

echo
echo "============================================================"
echo "Variable-generation submission"
echo "Station:             ${STATION}"
echo "Mode:                ${DATA_MODE}"
echo "FULL:                ${FULL}"
echo "Input directory:     ${IN_DIR}"
echo "Output directory:    ${OUT_DIR}"
echo "Run list:            ${RUNLIST}"
echo "Task map:            ${TASK_MAP}"
echo "Tracking directory:  ${TRACKING_DIR}"
echo "Number of runs:      ${NRUNS}"
echo "Runs per task:       ${CHUNK}"
echo "Array tasks:         ${NTASKS}"
echo "Array range:         0-$((NTASKS - 1))"
echo "Job name:            ${JOB_NAME}"
echo "Slurm output:        ${SLURM_OUT}"
echo "Slurm error:         ${SLURM_ERR}"

if [[ -n "${EXCLUDE_NODES}" ]]; then
    echo "Excluded nodes:      ${EXCLUDE_NODES}"
else
    echo "Excluded nodes:      none"
fi

echo "============================================================"


echo
echo "First ten runs:"

head -n 10 "${RUNLIST}"


# ============================================================
# Save submission configuration
# ============================================================

cat > "${TRACKING_DIR}/submission_config.txt" << EOF
STATION=${STATION}
FULL=${FULL}
DATA_MODE=${DATA_MODE}
CHUNK=${CHUNK}
NRUNS=${NRUNS}
NTASKS=${NTASKS}
IN_DIR=${IN_DIR}
OUT_DIR=${OUT_DIR}
RUNLIST=${RUNLIST}
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

FINALIZE_SCRIPT="${TRACKING_DIR}/update_unprocessed_runs.sh"


cat > "${FINALIZE_SCRIPT}" << EOF
#!/bin/bash

EXPECTED_RUNS="${RUNLIST}"

PROCESSED_DIR="${TRACKING_DIR}/processed"

PROCESSED_RUNS="${TRACKING_DIR}/processed_runs.txt"

UNPROCESSED_RUNS="${TRACKING_DIR}/unprocessed_runs.txt"


find "\${PROCESSED_DIR}" \
    -maxdepth 1 \
    -type f \
    -name "task_*.txt" \
    -exec cat {} + 2>/dev/null \
    | awk '
        NF == 1 &&
        \$1 ~ /^[0-9]+\$/ {

            if (!seen[\$1]++) {
                print \$1
            }
        }
    ' \
    | sort -n \
    > "\${PROCESSED_RUNS}"


awk '
    NR == FNR {
        processed[\$1] = 1
        next
    }

    !(\$1 in processed) {
        print \$1
    }
' "\${PROCESSED_RUNS}" \
  "\${EXPECTED_RUNS}" \
  | sort -n \
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
echo "Submitting Slurm array:"
echo "0-$((NTASKS - 1))"


if [[ -n "${EXCLUDE_NODES}" ]]; then

    SBATCH_RESULT=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --output="${SLURM_OUT}" \
        --error="${SLURM_ERR}" \
        --exclude="${EXCLUDE_NODES}" \
        --array="0-$((NTASKS - 1))" \
        --export="ALL,STATION=${STATION},FULL=${FULL},DATA_MODE=${DATA_MODE},CHUNK=${CHUNK},IN_DIR=${IN_DIR},OUT_DIR=${OUT_DIR},RUNLIST=${RUNLIST},TRACKING_DIR=${TRACKING_DIR}" \
        "${SLURM_SCRIPT}")

else

    SBATCH_RESULT=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --output="${SLURM_OUT}" \
        --error="${SLURM_ERR}" \
        --array="0-$((NTASKS - 1))" \
        --export="ALL,STATION=${STATION},FULL=${FULL},DATA_MODE=${DATA_MODE},CHUNK=${CHUNK},IN_DIR=${IN_DIR},OUT_DIR=${OUT_DIR},RUNLIST=${RUNLIST},TRACKING_DIR=${TRACKING_DIR}" \
        "${SLURM_SCRIPT}")

fi


SBATCH_STATUS=$?


if (( SBATCH_STATUS != 0 )); then
    echo "ERROR: Slurm submission failed."
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
    --job-name="finalize_vars_${DATA_MODE}_s${STATION}" \
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
echo "Variable-generation array submitted."
echo "Array job ID:       ${ARRAY_JOB_ID}"
echo "Station:            ${STATION}"
echo "Mode:               ${DATA_MODE}"
echo "Number of runs:     ${NRUNS}"
echo "Runs per task:      ${CHUNK}"
echo "Number of tasks:    ${NTASKS}"
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
