#!/bin/bash


# ============================================================
# User configuration
# ============================================================

STATION="13"

# Number of simulation files processed by each array task.
# Use CHUNK="1" for one run per task.
CHUNK="10"

ENERGIES=(
    16.0
    16.5
    17.0
    17.5
    18.0
    18.5
    19.0
)

BASE_DIR="/mnt/nrdstor/hep/martinliu/data/simData/CR/project_LDA/simPreprocessed/filteredData/station${STATION}"

OUT_DIR="/mnt/nrdstor/hep/martinliu/data/simData/CR/project_LDA/simPreprocessed/variableFiles/station${STATION}"

TRACKING_ROOT="/home/hep/martinliu/research/project_LDA/tracking_runs"

SLURM_SCRIPT="/home/hep/martinliu/research/project_LDA/makeVariables_CR_proxy.slurm"


# Nodes that have repeatedly produced hanging jobs.
#
# Set to an empty string to disable node exclusion:
#
# EXCLUDE_NODES=""
#
EXCLUDE_NODES="c1424,c2005,c1213,c1406,c1908,c1926,c2015,c2018,c2009,c1514,c1921,c2021"


# Dynamic Slurm naming.
JOB_NAME="vars_proxy_s${STATION}"

SLURM_OUT="/work/hep/martinliu/slurm_out/${JOB_NAME}-%A_%a.out"

SLURM_ERR="/work/hep/martinliu/slurm_out/${JOB_NAME}-%A_%a.err"


# ============================================================
# Optional rerun list
# ============================================================
#
# Normal submission:
#
#     bash makeVariables_CR_proxy.sh
#
# Rerun only previously unprocessed runs:
#
#     bash makeVariables_CR_proxy.sh \
#         /path/to/unprocessed_runs.txt
#
# The rerun list must contain:
#
#     ENERGY RUN
#
# Example:
#
#     16.0 1574
#     17.5 16
#     18.0 52
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
    echo "ERROR: Input base directory does not exist:"
    echo "${BASE_DIR}"
    exit 1
fi


if [[ ! -f "${SLURM_SCRIPT}" ]]; then
    echo "ERROR: Slurm script does not exist:"
    echo "${SLURM_SCRIPT}"
    exit 1
fi


# ============================================================
# Create unique tracking directory
# ============================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")


if [[ -n "${SOURCE_RUN_LIST}" ]]; then

    TRACKING_DIR="${TRACKING_ROOT}/makeVariables_proxy_s${STATION}_rerun_${TIMESTAMP}"

else

    TRACKING_DIR="${TRACKING_ROOT}/makeVariables_proxy_s${STATION}_${TIMESTAMP}"

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


TASK_LIST="${TRACKING_DIR}/expected_runs.txt"

TASK_LIST_TMP="${TRACKING_DIR}/expected_runs.txt.tmp"

: > "${TASK_LIST_TMP}"


# ============================================================
# Create expected-run list
# ============================================================
#
# Each line contains:
#
#     ENERGY RUN ABSOLUTE_INPUT_FILE
# ============================================================

if [[ -n "${SOURCE_RUN_LIST}" ]]; then

    echo "Creating rerun task list from:"
    echo "${SOURCE_RUN_LIST}"


    if [[ ! -s "${SOURCE_RUN_LIST}" ]]; then
        echo "ERROR: Rerun list does not exist or is empty:"
        echo "${SOURCE_RUN_LIST}"
        exit 1
    fi


    while read -r ENERGY RUN EXTRA; do

        if [[ -z "${ENERGY}" && -z "${RUN}" ]]; then
            continue
        fi


        if [[ -z "${ENERGY}" || -z "${RUN}" || -n "${EXTRA}" ]]; then
            echo "ERROR: Malformed rerun-list entry:"
            echo "${ENERGY} ${RUN} ${EXTRA}"
            exit 1
        fi


        if [[ ! "${ENERGY}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            echo "ERROR: Invalid energy in rerun list:"
            echo "${ENERGY}"
            exit 1
        fi


        if [[ ! "${RUN}" =~ ^[0-9]+$ ]]; then
            echo "ERROR: Invalid run number in rerun list:"
            echo "${RUN}"
            exit 1
        fi


        RUN=$((10#${RUN}))


        DIR_IN="${BASE_DIR}/lgE${ENERGY}"


        if [[ ! -d "${DIR_IN}" ]]; then
            echo "ERROR: Input energy directory does not exist:"
            echo "${DIR_IN}"
            exit 1
        fi


        INPUT_FILE=""


        while IFS= read -r FILE; do

            FILE_RUN_STRING=$(basename "${FILE}" \
                | sed -nE \
                    "s/^filtered_sim_s${STATION}_${ENERGY}eV_r([0-9]+)\.root$/\1/p")


            if [[ -z "${FILE_RUN_STRING}" ]]; then
                continue
            fi


            if [[ ! "${FILE_RUN_STRING}" =~ ^[0-9]+$ ]]; then
                continue
            fi


            FILE_RUN=$((10#${FILE_RUN_STRING}))


            if (( FILE_RUN == RUN )); then
                INPUT_FILE="${FILE}"
                break
            fi

        done < <(
            find "${DIR_IN}" \
                -maxdepth 1 \
                -type f \
                -size +0c \
                -name "filtered_sim_s${STATION}_${ENERGY}eV_r*.root" \
                -print \
                | sort -V
        )


        if [[ -z "${INPUT_FILE}" ]]; then
            echo "ERROR: Could not find filtered simulation file."
            echo "Energy: ${ENERGY}"
            echo "Run:    ${RUN}"
            exit 1
        fi


        echo "${ENERGY} ${RUN} ${INPUT_FILE}" \
            >> "${TASK_LIST_TMP}"

    done < "${SOURCE_RUN_LIST}"


else

    echo "Scanning all filtered simulation files."


    for ENERGY in "${ENERGIES[@]}"; do

        DIR_IN="${BASE_DIR}/lgE${ENERGY}"


        if [[ ! -d "${DIR_IN}" ]]; then
            echo "WARNING: Input directory does not exist:"
            echo "${DIR_IN}"
            continue
        fi


        N_FILES=0


        while IFS= read -r FILE; do

            FILE_RUN_STRING=$(basename "${FILE}" \
                | sed -nE \
                    "s/^filtered_sim_s${STATION}_${ENERGY}eV_r([0-9]+)\.root$/\1/p")


            if [[ -z "${FILE_RUN_STRING}" ]]; then
                continue
            fi


            if [[ ! "${FILE_RUN_STRING}" =~ ^[0-9]+$ ]]; then
                echo "WARNING: Invalid run number extracted from:"
                echo "${FILE}"
                continue
            fi


            RUN=$((10#${FILE_RUN_STRING}))


            echo "${ENERGY} ${RUN} ${FILE}" \
                >> "${TASK_LIST_TMP}"


            N_FILES=$((N_FILES + 1))

        done < <(
            find "${DIR_IN}" \
                -maxdepth 1 \
                -type f \
                -size +0c \
                -name "filtered_sim_s${STATION}_${ENERGY}eV_r*.root" \
                -print \
                | sort -V
        )


        echo "ENERGY=${ENERGY}, N_FILES=${N_FILES}"

    done

fi


# ============================================================
# Sort and deduplicate
# ============================================================

sort -k1,1n -k2,2n -u \
    "${TASK_LIST_TMP}" \
    > "${TASK_LIST}"

SORT_STATUS=$?


rm -f "${TASK_LIST_TMP}"


if (( SORT_STATUS != 0 )); then
    echo "ERROR: Could not create sorted task list."
    exit 1
fi


N_RUNS=$(wc -l < "${TASK_LIST}")

N_RUNS=$(printf '%s' "${N_RUNS}" | tr -d '[:space:]')


if [[ ! "${N_RUNS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Could not determine number of simulation runs."
    exit 1
fi


if (( N_RUNS == 0 )); then
    echo "ERROR: No matching filtered simulation files were found."
    exit 1
fi


N_TASKS=$(( (N_RUNS + CHUNK - 1) / CHUNK ))


# ============================================================
# Create task map
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
# Print submission summary
# ============================================================

echo
echo "============================================================"
echo "Proxy simulation variable generation"
echo "Station:             ${STATION}"
echo "Input base:          ${BASE_DIR}"
echo "Output directory:    ${OUT_DIR}"
echo "Tracking directory:  ${TRACKING_DIR}"
echo "Task list:           ${TASK_LIST}"
echo "Task map:            ${TASK_MAP}"
echo "Number of runs:      ${N_RUNS}"
echo "Runs per task:       ${CHUNK}"
echo "Array tasks:         ${N_TASKS}"
echo "Array range:         0-$((N_TASKS - 1))"
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
' "${TASK_LIST}" | sort -k1,1n


echo
echo "First ten entries:"

head -n 10 "${TASK_LIST}"


# ============================================================
# Save submission configuration
# ============================================================

cat > "${TRACKING_DIR}/submission_config.txt" << EOF
STATION=${STATION}
CHUNK=${CHUNK}
ENERGIES=${ENERGIES[*]}
BASE_DIR=${BASE_DIR}
OUT_DIR=${OUT_DIR}
TASK_LIST=${TASK_LIST}
TASK_MAP=${TASK_MAP}
TRACKING_DIR=${TRACKING_DIR}
SOURCE_RUN_LIST=${SOURCE_RUN_LIST}
SLURM_SCRIPT=${SLURM_SCRIPT}
EXCLUDE_NODES=${EXCLUDE_NODES}
N_RUNS=${N_RUNS}
N_TASKS=${N_TASKS}
JOB_NAME=${JOB_NAME}
SLURM_OUT=${SLURM_OUT}
SLURM_ERR=${SLURM_ERR}
SUBMISSION_TIME=$(date --iso-8601=seconds)
EOF


# ============================================================
# Create finalizer script
# ============================================================

FINALIZE_SCRIPT="${TRACKING_DIR}/update_unprocessed_runs.sh"


cat > "${FINALIZE_SCRIPT}" << EOF
#!/bin/bash

EXPECTED_RUNS="${TASK_LIST}"

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
# Submit variable-generation array
# ============================================================

echo
echo "Submitting Slurm array:"
echo "0-$((N_TASKS - 1))"


if [[ -n "${EXCLUDE_NODES}" ]]; then

    SBATCH_RESULT=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --output="${SLURM_OUT}" \
        --error="${SLURM_ERR}" \
        --exclude="${EXCLUDE_NODES}" \
        --array="0-$((N_TASKS - 1))" \
        --export="ALL,STATION=${STATION},CHUNK=${CHUNK},OUT_DIR=${OUT_DIR},TASK_LIST=${TASK_LIST},TRACKING_DIR=${TRACKING_DIR}" \
        "${SLURM_SCRIPT}")

else

    SBATCH_RESULT=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --output="${SLURM_OUT}" \
        --error="${SLURM_ERR}" \
        --array="0-$((N_TASKS - 1))" \
        --export="ALL,STATION=${STATION},CHUNK=${CHUNK},OUT_DIR=${OUT_DIR},TASK_LIST=${TASK_LIST},TRACKING_DIR=${TRACKING_DIR}" \
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
    --job-name="finalize_vars_proxy_s${STATION}" \
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

    echo "Run manually after the array ends:"

    echo
    echo "bash ${FINALIZE_SCRIPT}"

fi


# ============================================================
# Final summary
# ============================================================

echo
echo "============================================================"
echo "Simulation variable job submitted."
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
