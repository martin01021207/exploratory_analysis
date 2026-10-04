#!/bin/bash

# These are fallback values.
# applyHitFilter_chunks.sh overrides the job name, output,
# and error paths when submitting the job.
#SBATCH --job-name=filter_real
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=18G
#SBATCH --time=48:00:00
#SBATCH --error=/work/hep/martinliu/slurm_out/filter_real-%A_%a.err
#SBATCH --output=/work/hep/martinliu/slurm_out/filter_real-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=martin01021207@huskers.unl.edu

# Placeholder; overridden during submission.
#SBATCH --array=0-0


# ============================================================
# Validate exported configuration
# ============================================================

REQUIRED_VARIABLES=(
    STATION
    FULL
    DATA_MODE
    CHUNK
    DIR_OUT
    JSON_DIR
    RUN_LIST
    TRACKING_DIR
)


for VARIABLE in "${REQUIRED_VARIABLES[@]}"; do

    if [[ -z "${!VARIABLE:-}" ]]; then

        echo "ERROR: Required variable ${VARIABLE} was not exported."

        exit 1

    fi

done


if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not defined."
    exit 1
fi


if [[ ! "${STATION}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid station number: ${STATION}"
    exit 1
fi


if [[ "${FULL}" != "true" && "${FULL}" != "false" ]]; then
    echo "ERROR: FULL must be either true or false."
    exit 1
fi


if [[ "${DATA_MODE}" != "full" && "${DATA_MODE}" != "burn" ]]; then
    echo "ERROR: DATA_MODE must be either full or burn."
    exit 1
fi


if [[ "${FULL}" == "true" && "${DATA_MODE}" != "full" ]]; then
    echo "ERROR: FULL=true requires DATA_MODE=full."
    exit 1
fi


if [[ "${FULL}" == "false" && "${DATA_MODE}" != "burn" ]]; then
    echo "ERROR: FULL=false requires DATA_MODE=burn."
    exit 1
fi


if [[ ! "${CHUNK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Invalid chunk size: ${CHUNK}"
    exit 1
fi


if [[ ! -s "${RUN_LIST}" ]]; then

    echo "ERROR: Run list does not exist or is empty:"
    echo "${RUN_LIST}"

    exit 1

fi


# ============================================================
# Tracking files
# ============================================================

mkdir -p "${TRACKING_DIR}/processed"
mkdir -p "${TRACKING_DIR}/failed"
mkdir -p "${TRACKING_DIR}/input_missing"
mkdir -p "${TRACKING_DIR}/output_missing"
mkdir -p "${TRACKING_DIR}/task_status"


RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"

HOSTNAME_CURRENT=$(hostname)


PROCESSED_FILE="${TRACKING_DIR}/processed/task_${SLURM_ARRAY_TASK_ID}_attempt_${RESTART_COUNT}.txt"

FAILED_FILE="${TRACKING_DIR}/failed/task_${SLURM_ARRAY_TASK_ID}_attempt_${RESTART_COUNT}.txt"

INPUT_MISSING_FILE="${TRACKING_DIR}/input_missing/task_${SLURM_ARRAY_TASK_ID}_attempt_${RESTART_COUNT}.txt"

OUTPUT_MISSING_FILE="${TRACKING_DIR}/output_missing/task_${SLURM_ARRAY_TASK_ID}_attempt_${RESTART_COUNT}.txt"

TASK_STATUS_FILE="${TRACKING_DIR}/task_status/task_${SLURM_ARRAY_TASK_ID}_attempt_${RESTART_COUNT}.txt"


: > "${PROCESSED_FILE}"
: > "${FAILED_FILE}"
: > "${INPUT_MISSING_FILE}"
: > "${OUTPUT_MISSING_FILE}"


{
    echo "status=STARTED"
    echo "job_id=${SLURM_JOB_ID}"
    echo "array_job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
    echo "restart_count=${RESTART_COUNT}"
    echo "hostname=${HOSTNAME_CURRENT}"
    echo "start_time=$(date --iso-8601=seconds)"
} > "${TASK_STATUS_FILE}"


# ============================================================
# Task information
# ============================================================

echo "============================================================"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Job name:        ${SLURM_JOB_NAME:-not_set}"
echo "Array job ID:    ${SLURM_ARRAY_JOB_ID:-not_set}"
echo "Array task ID:   ${SLURM_ARRAY_TASK_ID}"
echo "Restart count:   ${RESTART_COUNT}"
echo "Hostname:        ${HOSTNAME_CURRENT}"
echo "Start time:      $(date)"
echo "Station:         ${STATION}"
echo "Mode:            ${DATA_MODE}"
echo "FULL:            ${FULL}"
echo "Chunk size:      ${CHUNK}"
echo "Run list:        ${RUN_LIST}"
echo "Tracking dir:    ${TRACKING_DIR}"
echo "Output dir:      ${DIR_OUT}"
echo "============================================================"


# ============================================================
# Determine assigned runs
# ============================================================

START_LINE=$((SLURM_ARRAY_TASK_ID * CHUNK + 1))

END_LINE=$((START_LINE + CHUNK - 1))


mapfile -t TASK_LINES < <(
    sed -n "${START_LINE},${END_LINE}p" "${RUN_LIST}"
)


if (( ${#TASK_LINES[@]} == 0 )); then

    echo "ERROR: No runs assigned to array task ${SLURM_ARRAY_TASK_ID}."

    {
        echo "status=NO_RUNS_ASSIGNED"
        echo "end_time=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"

    exit 1

fi


echo
echo "Assigned run-list lines: ${START_LINE}-${END_LINE}"

echo "Number of assigned runs: ${#TASK_LINES[@]}"

printf '  %s\n' "${TASK_LINES[@]}"


# ============================================================
# Helper: record assigned runs as failed
# ============================================================

record_assigned_runs_as_failed()
{
    printf '%s\n' "${TASK_LINES[@]}" \
        | awk '
            NF >= 2 {
                print $1, $2
            }
        ' \
        >> "${FAILED_FILE}"
}


# ============================================================
# Enter project directory
# ============================================================

cd /home/hep/martinliu/research/project_LDA/

CD_STATUS=$?


if (( CD_STATUS != 0 )); then

    echo "ERROR: Could not enter project directory."

    record_assigned_runs_as_failed


    {
        echo "status=PROJECT_DIRECTORY_FAILED"
        echo "cd_exit_code=${CD_STATUS}"
        echo "end_time=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"

    exit 1

fi


echo
echo "Project directory: $(pwd)"


# ============================================================
# Set up software environment
# ============================================================

echo
echo "Starting environment setup at $(date)"
echo "Node: ${HOSTNAME_CURRENT}"
echo "Sourcing CVMFS environment at $(date)"


source /cvmfs/rnog.opensciencegrid.org/software/trunk/setup.sh

CVMFS_STATUS=$?


if (( CVMFS_STATUS != 0 )); then

    echo "ERROR: CVMFS setup failed with exit code ${CVMFS_STATUS}."
    echo "Node: ${HOSTNAME_CURRENT}"

    record_assigned_runs_as_failed


    {
        echo "status=CVMFS_SETUP_FAILED"
        echo "cvmfs_exit_code=${CVMFS_STATUS}"
        echo "hostname=${HOSTNAME_CURRENT}"
        echo "end_time=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"

    exit 1

fi


echo "CVMFS setup completed successfully at $(date)"


module unload python/3.9 2>/dev/null

MODULE_STATUS=$?


if (( MODULE_STATUS != 0 )); then
    echo "NOTE: python/3.9 was not loaded or could not be unloaded."
fi


echo "Activating virtual environment at $(date)"


source /home/hep/martinliu/software/env_rnog/bin/activate

VENV_STATUS=$?


if (( VENV_STATUS != 0 )); then

    echo "ERROR: Virtual environment activation failed."

    record_assigned_runs_as_failed


    {
        echo "status=VENV_ACTIVATION_FAILED"
        echo "venv_exit_code=${VENV_STATUS}"
        echo "end_time=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"

    exit 1

fi


PYTHON_EXECUTABLE=$(command -v python)

COMMAND_STATUS=$?


if (( COMMAND_STATUS != 0 )) || [[ -z "${PYTHON_EXECUTABLE}" ]]; then

    echo "ERROR: Python was not found after environment setup."

    record_assigned_runs_as_failed


    {
        echo "status=PYTHON_NOT_FOUND"
        echo "end_time=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"

    exit 1

fi


echo "Virtual environment activated successfully."

echo "Python executable: ${PYTHON_EXECUTABLE}"


python --version

PYTHON_VERSION_STATUS=$?


if (( PYTHON_VERSION_STATUS != 0 )); then

    echo "ERROR: Python exists but could not be executed."

    record_assigned_runs_as_failed


    {
        echo "status=PYTHON_EXECUTION_FAILED"
        echo "python_exit_code=${PYTHON_VERSION_STATUS}"
        echo "end_time=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"

    exit 1

fi


echo "Environment setup completed at $(date)"


# ============================================================
# Input/output directories
# ============================================================

DIR_IN="/mnt/nrdstor/hep/martinliu/data/rnogData/"


if [[ ! -d "${DIR_IN}" ]]; then

    echo "ERROR: Input directory does not exist:"
    echo "${DIR_IN}"

    record_assigned_runs_as_failed


    {
        echo "status=INPUT_DIRECTORY_MISSING"
        echo "end_time=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"

    exit 1

fi


mkdir -p "${DIR_OUT}"

MKDIR_STATUS=$?


if (( MKDIR_STATUS != 0 )) || [[ ! -d "${DIR_OUT}" ]]; then

    echo "ERROR: Output directory could not be created:"
    echo "${DIR_OUT}"

    record_assigned_runs_as_failed


    {
        echo "status=OUTPUT_DIRECTORY_FAILED"
        echo "mkdir_exit_code=${MKDIR_STATUS}"
        echo "end_time=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"

    exit 1

fi


echo
echo "Input directory:  ${DIR_IN}"
echo "Output directory: ${DIR_OUT}"


# ============================================================
# Process assigned runs
# ============================================================

N_PROCESSED=0

N_FAILED=0

N_INPUT_MISSING=0

N_OUTPUT_MISSING=0


for TASK_LINE in "${TASK_LINES[@]}"; do

    TASK_LINE=$(printf '%s' "${TASK_LINE}" \
        | tr -d '\r' \
        | xargs)


    if [[ -z "${TASK_LINE}" ]]; then

        echo "WARNING: Empty task-list line."

        continue

    fi


    YEAR=""

    RUN=""

    EXTRA=""


    read -r YEAR RUN EXTRA <<< "${TASK_LINE}"


    if [[ -z "${YEAR}" || -z "${RUN}" || -n "${EXTRA}" ]]; then

        echo "ERROR: Malformed task-list entry:"
        echo "${TASK_LINE}"

        echo "${TASK_LINE}" \
            >> "${FAILED_FILE}"

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    if [[ ! "${YEAR}" =~ ^[0-9]{4}$ ]]; then

        echo "ERROR: Invalid year:"
        echo "${TASK_LINE}"

        echo "${TASK_LINE}" \
            >> "${FAILED_FILE}"

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    if [[ ! "${RUN}" =~ ^[0-9]+$ ]]; then

        echo "ERROR: Invalid run:"
        echo "${TASK_LINE}"

        echo "${TASK_LINE}" \
            >> "${FAILED_FILE}"

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    JSON="${JSON_DIR}/station${STATION}_${YEAR}_burn_sample_evt_num_dict.json"

    RAW_RUN_DIR="${DIR_IN}/station${STATION}/run${RUN}"

    EXPECTED_OUTPUT="${DIR_OUT}/filtered_s${STATION}_r${RUN}.root"


    echo
    echo "------------------------------------------------------------"
    echo "Processing station ${STATION}"
    echo "Year:   ${YEAR}"
    echo "Run:    ${RUN}"
    echo "Node:   ${HOSTNAME_CURRENT}"
    echo "Start:  $(date)"
    echo "JSON:   ${JSON}"
    echo "Mode:   ${DATA_MODE}"
    echo "Input:  ${RAW_RUN_DIR}"
    echo "Output: ${EXPECTED_OUTPUT}"
    echo "------------------------------------------------------------"


    # Record the exact run this task is currently processing.
    {
        echo "current_year=${YEAR}"
        echo "current_run=${RUN}"
        echo "current_run_start=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"


    # ========================================================
    # Validate JSON
    # ========================================================

    if [[ ! -s "${JSON}" ]]; then

        echo "ERROR: JSON file is missing or empty:"
        echo "${JSON}"

        echo "${YEAR} ${RUN}" \
            >> "${FAILED_FILE}"

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    # ========================================================
    # Validate raw-data run directory
    # ========================================================

    if [[ ! -d "${RAW_RUN_DIR}" ]]; then

        echo "ERROR: Raw-data run directory does not exist:"
        echo "${RAW_RUN_DIR}"

        echo "${YEAR} ${RUN}" \
            >> "${INPUT_MISSING_FILE}"

        echo "${YEAR} ${RUN}" \
            >> "${FAILED_FILE}"

        N_INPUT_MISSING=$((N_INPUT_MISSING + 1))

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    # ========================================================
    # Run Python
    # ========================================================

    if [[ "${FULL}" == "true" ]]; then

        echo "Running in full-data mode."
        echo "Burn-sample events will be excluded."


        python -u applyHitFilter.py \
            "${DIR_IN}" \
            "${DIR_OUT}" \
            "${STATION}" \
            "${RUN}" \
            --json_select "${JSON}" \
            --isExcluded

    else

        echo "Running in burn-data mode."
        echo "Only burn-sample events selected by the JSON will be processed."


        python -u applyHitFilter.py \
            "${DIR_IN}" \
            "${DIR_OUT}" \
            "${STATION}" \
            "${RUN}" \
            --json_select "${JSON}"

    fi


    PYTHON_STATUS=$?


    # ========================================================
    # Check Python result
    # ========================================================

    if (( PYTHON_STATUS != 0 )); then

        echo "ERROR: Run ${RUN} failed."
        echo "Python exit code: ${PYTHON_STATUS}"

        echo "${YEAR} ${RUN}" \
            >> "${FAILED_FILE}"

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    # ========================================================
    # Verify expected output
    # ========================================================

    if [[ -s "${EXPECTED_OUTPUT}" ]]; then

        # IMPORTANT:
        # Record this run immediately so that if the task hangs
        # on a later run, this completed run remains tracked.
        echo "${YEAR} ${RUN}" \
            >> "${PROCESSED_FILE}"


        N_PROCESSED=$((N_PROCESSED + 1))


        {
            echo "completed_year=${YEAR}"
            echo "completed_run=${RUN}"
            echo "completed_run_time=$(date --iso-8601=seconds)"
        } >> "${TASK_STATUS_FILE}"


        echo "Run ${RUN} completed successfully."

        echo "Output:"
        echo "${EXPECTED_OUTPUT}"


    else

        echo "ERROR: Python returned success, but the expected output is missing or empty."

        echo "${YEAR} ${RUN}" \
            >> "${OUTPUT_MISSING_FILE}"

        echo "${YEAR} ${RUN}" \
            >> "${FAILED_FILE}"

        N_OUTPUT_MISSING=$((N_OUTPUT_MISSING + 1))

        N_FAILED=$((N_FAILED + 1))

    fi


    echo "End: $(date)"

done


# ============================================================
# Final task status
# ============================================================

if (( N_FAILED > 0 )); then

    FINAL_STATUS="FINISHED_WITH_FAILURES"

else

    FINAL_STATUS="FINISHED"

fi


{
    echo "status=${FINAL_STATUS}"
    echo "processed=${N_PROCESSED}"
    echo "failed=${N_FAILED}"
    echo "input_missing=${N_INPUT_MISSING}"
    echo "output_missing=${N_OUTPUT_MISSING}"
    echo "end_time=$(date --iso-8601=seconds)"
} >> "${TASK_STATUS_FILE}"


echo
echo "============================================================"
echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
echo "Node:                    ${HOSTNAME_CURRENT}"
echo "Successfully processed:  ${N_PROCESSED}"
echo "Failed:                  ${N_FAILED}"
echo "Missing inputs:          ${N_INPUT_MISSING}"
echo "Missing outputs:         ${N_OUTPUT_MISSING}"
echo "Processed-run file:      ${PROCESSED_FILE}"
echo "Failed-run file:         ${FAILED_FILE}"
echo "Input-missing file:      ${INPUT_MISSING_FILE}"
echo "Output-missing file:     ${OUTPUT_MISSING_FILE}"
echo "Task-status file:        ${TASK_STATUS_FILE}"
echo "============================================================"


if (( N_FAILED > 0 )); then
    exit 1
fi


if (( N_PROCESSED == 0 )); then

    echo "ERROR: This array task did not successfully process any runs."

    exit 1

fi


exit 0
