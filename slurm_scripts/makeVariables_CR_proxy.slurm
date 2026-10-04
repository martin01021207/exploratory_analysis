#!/bin/bash

# Fallback values.
# makeVariables_CR_proxy.sh overrides the job name and log paths
# with sbatch command-line options.
#SBATCH --job-name=vars_proxy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=48:00:00
#SBATCH --error=/work/hep/martinliu/slurm_out/vars_proxy-%A_%a.err
#SBATCH --output=/work/hep/martinliu/slurm_out/vars_proxy-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=martin01021207@huskers.unl.edu

# Placeholder; overridden during submission.
#SBATCH --array=0-0


# ============================================================
# Validate exported configuration
# ============================================================

REQUIRED_VARIABLES=(
    STATION
    CHUNK
    OUT_DIR
    TASK_LIST
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


if [[ ! "${CHUNK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Invalid chunk size: ${CHUNK}"
    exit 1
fi


if [[ ! -s "${TASK_LIST}" ]]; then
    echo "ERROR: Task list does not exist or is empty:"
    echo "${TASK_LIST}"
    exit 1
fi


# ============================================================
# Create task tracking files
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
    echo "job_name=${SLURM_JOB_NAME:-not_set}"
    echo "array_job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
    echo "restart_count=${RESTART_COUNT}"
    echo "hostname=${HOSTNAME_CURRENT}"
    echo "start_time=$(date --iso-8601=seconds)"
} > "${TASK_STATUS_FILE}"


# ============================================================
# Print task information
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
echo "Chunk size:      ${CHUNK}"
echo "Task list:       ${TASK_LIST}"
echo "Tracking dir:    ${TRACKING_DIR}"
echo "Output dir:      ${OUT_DIR}"
echo "============================================================"


# ============================================================
# Read task-list entries assigned to this array task
# ============================================================

START_LINE=$((SLURM_ARRAY_TASK_ID * CHUNK + 1))

END_LINE=$((START_LINE + CHUNK - 1))


mapfile -t TASK_LINES < <(
    sed -n "${START_LINE},${END_LINE}p" "${TASK_LIST}"
)


if (( ${#TASK_LINES[@]} == 0 )); then

    echo "ERROR: No simulation runs were assigned to this task."

    {
        echo "status=NO_RUNS_ASSIGNED"
        echo "end_time=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"

    exit 1

fi


echo
echo "Assigned task-list lines: ${START_LINE}-${END_LINE}"

echo "Number of assigned runs: ${#TASK_LINES[@]}"

printf '  %s\n' "${TASK_LINES[@]}"


# ============================================================
# Helper for recording all assigned runs as failed
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
echo "Working directory: $(pwd)"


# ============================================================
# Set up software environment
# ============================================================

echo
echo "Sourcing CVMFS environment at $(date)"
echo "Node: ${HOSTNAME_CURRENT}"


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


echo "Activating env_rnog at $(date)"


source /home/hep/martinliu/software/env_rnog/bin/activate

VENV_STATUS=$?


if (( VENV_STATUS != 0 )); then

    echo "ERROR: Failed to activate env_rnog."

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

    echo "ERROR: Python could not be found after environment setup."

    record_assigned_runs_as_failed


    {
        echo "status=PYTHON_NOT_FOUND"
        echo "end_time=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"

    exit 1

fi


echo "Python executable: ${PYTHON_EXECUTABLE}"


python --version

PYTHON_VERSION_STATUS=$?


if (( PYTHON_VERSION_STATUS != 0 )); then

    echo "ERROR: Python could not be executed."

    record_assigned_runs_as_failed


    {
        echo "status=PYTHON_EXECUTION_FAILED"
        echo "python_exit_code=${PYTHON_VERSION_STATUS}"
        echo "end_time=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"

    exit 1

fi


# ============================================================
# Create output directory
# ============================================================

mkdir -p "${OUT_DIR}"

MKDIR_STATUS=$?


if (( MKDIR_STATUS != 0 )) || [[ ! -d "${OUT_DIR}" ]]; then

    echo "ERROR: Could not create output directory:"
    echo "${OUT_DIR}"

    record_assigned_runs_as_failed


    {
        echo "status=OUTPUT_DIRECTORY_FAILED"
        echo "mkdir_exit_code=${MKDIR_STATUS}"
        echo "end_time=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"

    exit 1

fi


# ============================================================
# Process assigned simulation runs
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


    ENERGY=""

    RUN=""

    IN_FILE=""

    EXTRA=""


    read -r ENERGY RUN IN_FILE EXTRA <<< "${TASK_LINE}"


    if [[ -z "${ENERGY}" || -z "${RUN}" || -z "${IN_FILE}" || -n "${EXTRA}" ]]; then

        echo "ERROR: Malformed task-list entry:"
        echo "${TASK_LINE}"

        echo "${ENERGY} ${RUN}" \
            >> "${FAILED_FILE}"

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    if [[ ! "${ENERGY}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then

        echo "ERROR: Invalid energy:"
        echo "${ENERGY}"

        echo "${ENERGY} ${RUN}" \
            >> "${FAILED_FILE}"

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    if [[ ! "${RUN}" =~ ^[0-9]+$ ]]; then

        echo "ERROR: Invalid run number:"
        echo "${RUN}"

        echo "${ENERGY} ${RUN}" \
            >> "${FAILED_FILE}"

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    if [[ "${IN_FILE}" != /* ]]; then

        echo "ERROR: Input path is not absolute:"
        echo "${IN_FILE}"

        echo "${ENERGY} ${RUN}" \
            >> "${FAILED_FILE}"

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    echo
    echo "------------------------------------------------------------"
    echo "Processing station ${STATION}"
    echo "Node:   ${HOSTNAME_CURRENT}"
    echo "Energy: ${ENERGY}"
    echo "Run:    ${RUN}"
    echo "Start:  $(date)"
    echo "Input:  ${IN_FILE}"
    echo "Output directory: ${OUT_DIR}"
    echo "------------------------------------------------------------"


    {
        echo "current_energy=${ENERGY}"
        echo "current_run=${RUN}"
        echo "current_run_start=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"


    # ========================================================
    # Validate input
    # ========================================================

    if [[ ! -s "${IN_FILE}" ]]; then

        echo "ERROR: Input file is missing or empty."

        echo "${ENERGY} ${RUN}" \
            >> "${INPUT_MISSING_FILE}"

        echo "${ENERGY} ${RUN}" \
            >> "${FAILED_FILE}"

        N_INPUT_MISSING=$((N_INPUT_MISSING + 1))

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    # ========================================================
    # Run makeVariables.py
    # ========================================================

    python -u makeVariables.py \
        "${IN_FILE}" \
        "${OUT_DIR}"

    PYTHON_STATUS=$?


    if (( PYTHON_STATUS != 0 )); then

        echo "ERROR: makeVariables.py failed."
        echo "Exit code: ${PYTHON_STATUS}"

        echo "${ENERGY} ${RUN}" \
            >> "${FAILED_FILE}"

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    # ========================================================
    # Verify output
    # ========================================================

    OUTPUT_FOUND=$(find "${OUT_DIR}" \
        -maxdepth 1 \
        -type f \
        -size +0c \
        -name "*.root" \
        -print 2>/dev/null \
        | grep -E "/[^/]*${ENERGY}[^/]*_r0*${RUN}\.root$" \
        | head -n 1)


    if [[ -n "${OUTPUT_FOUND}" ]]; then

        # Record this run immediately. If this task later hangs
        # on another run, the completed run remains recorded.
        echo "${ENERGY} ${RUN}" \
            >> "${PROCESSED_FILE}"


        N_PROCESSED=$((N_PROCESSED + 1))


        {
            echo "completed_energy=${ENERGY}"
            echo "completed_run=${RUN}"
            echo "completed_run_time=$(date --iso-8601=seconds)"
        } >> "${TASK_STATUS_FILE}"


        echo "Simulation run completed successfully."
        echo "Output:"
        echo "${OUTPUT_FOUND}"


    else

        echo "ERROR: Python returned successfully, but no matching output file was found."


        echo "${ENERGY} ${RUN}" \
            >> "${OUTPUT_MISSING_FILE}"

        echo "${ENERGY} ${RUN}" \
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
echo "Missing-input file:      ${INPUT_MISSING_FILE}"
echo "Missing-output file:     ${OUTPUT_MISSING_FILE}"
echo "Task-status file:        ${TASK_STATUS_FILE}"
echo "============================================================"


if (( N_FAILED > 0 )); then
    exit 1
fi


if (( N_PROCESSED == 0 )); then

    echo "ERROR: This array task did not process any runs."

    exit 1

fi


exit 0
