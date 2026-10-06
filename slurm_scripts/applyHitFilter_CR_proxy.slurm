#!/bin/bash

# Fallback values.
# applyHitFilter_CR_proxy.sh overrides the job name and log
# filenames using sbatch command-line options.
#SBATCH --job-name=filter_proxy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=48:00:00
#SBATCH --error=/work/hep/martinliu/slurm_out/filter_proxy-%A_%a.err
#SBATCH --output=/work/hep/martinliu/slurm_out/filter_proxy-%A_%a.out
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
    BASE_DIR
    DIR_OUT_BASE
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
# Create tracking files
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
    echo "chunk=${CHUNK}"
    echo "start_time=$(date --iso-8601=seconds)"
} > "${TASK_STATUS_FILE}"


# ============================================================
# Determine assigned energy/run pairs
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
echo "Assigned lines:  ${START_LINE}-${END_LINE}"
echo "Assigned runs:   ${#TASK_LINES[@]}"
echo "============================================================"

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
    EXTRA=""


    read -r ENERGY RUN EXTRA <<< "${TASK_LINE}"


    if [[ -z "${ENERGY}" || -z "${RUN}" || -n "${EXTRA}" ]]; then

        echo "ERROR: Malformed task-list entry:"
        echo "${TASK_LINE}"

        echo "${TASK_LINE}" \
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

        echo "ERROR: Invalid run:"
        echo "${RUN}"

        echo "${ENERGY} ${RUN}" \
            >> "${FAILED_FILE}"

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    DIR_IN="${BASE_DIR}/lgE${ENERGY}"

    DIR_OUT="${DIR_OUT_BASE}/lgE${ENERGY}"


    echo
    echo "------------------------------------------------------------"
    echo "Processing station ${STATION}"
    echo "Node:   ${HOSTNAME_CURRENT}"
    echo "Energy: ${ENERGY}"
    echo "Run:    ${RUN}"
    echo "Start:  $(date)"
    echo "Input directory:  ${DIR_IN}"
    echo "Output directory: ${DIR_OUT}"
    echo "------------------------------------------------------------"


    {
        echo "current_energy=${ENERGY}"
        echo "current_run=${RUN}"
        echo "current_run_start=$(date --iso-8601=seconds)"
    } >> "${TASK_STATUS_FILE}"


    # ========================================================
    # Validate input directory
    # ========================================================

    if [[ ! -d "${DIR_IN}" ]]; then

        echo "ERROR: Input directory does not exist:"
        echo "${DIR_IN}"

        echo "${ENERGY} ${RUN}" \
            >> "${INPUT_MISSING_FILE}"

        echo "${ENERGY} ${RUN}" \
            >> "${FAILED_FILE}"

        N_INPUT_MISSING=$((N_INPUT_MISSING + 1))
        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    # ========================================================
    # Locate matching input .nur file
    # ========================================================

    INPUT_FOUND=""


    while IFS= read -r INPUT_FILE; do

        RUN_STRING=$(basename "${INPUT_FILE}" \
            | sed -nE 's/.*_[jc]([0-9]+).*\.nur$/\1/p')


        if [[ -z "${RUN_STRING}" ]]; then
            continue
        fi


        if [[ ! "${RUN_STRING}" =~ ^[0-9]+$ ]]; then
            continue
        fi


        INPUT_RUN=$((10#${RUN_STRING}))


        if (( INPUT_RUN == RUN )); then

            INPUT_FOUND="${INPUT_FILE}"

            break

        fi

    done < <(
        find "${DIR_IN}" \
            -maxdepth 1 \
            -type f \
            -size +0c \
            -name "*.nur" \
            -print
    )


    if [[ -z "${INPUT_FOUND}" ]]; then

        echo "ERROR: No matching nonempty .nur input file was found."
        echo "Energy: ${ENERGY}"
        echo "Run:    ${RUN}"

        echo "${ENERGY} ${RUN}" \
            >> "${INPUT_MISSING_FILE}"

        echo "${ENERGY} ${RUN}" \
            >> "${FAILED_FILE}"

        N_INPUT_MISSING=$((N_INPUT_MISSING + 1))
        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    echo "Matching input file:"
    echo "${INPUT_FOUND}"


    # ========================================================
    # Create output directory
    # ========================================================

    mkdir -p "${DIR_OUT}"

    MKDIR_STATUS=$?


    if (( MKDIR_STATUS != 0 )) || [[ ! -d "${DIR_OUT}" ]]; then

        echo "ERROR: Could not create output directory:"
        echo "${DIR_OUT}"

        echo "${ENERGY} ${RUN}" \
            >> "${FAILED_FILE}"

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    # ========================================================
    # Run Python
    # ========================================================

    python -u applyHitFilter.py \
        "${DIR_IN}" \
        "${DIR_OUT}" \
        "${STATION}" \
        "${RUN}" \
        --isSim \
        --sim_E "${ENERGY}"


    PYTHON_STATUS=$?


    if (( PYTHON_STATUS != 0 )); then

        echo "ERROR: Python failed with exit code ${PYTHON_STATUS}."

        echo "${ENERGY} ${RUN}" \
            >> "${FAILED_FILE}"

        N_FAILED=$((N_FAILED + 1))

        continue

    fi


    # ========================================================
    # Verify output
    # ========================================================

    OUTPUT_FOUND=$(find "${DIR_OUT}" \
        -maxdepth 1 \
        -type f \
        -size +0c \
        -name "filtered_sim_s${STATION}_${ENERGY}eV_r*.root" \
        -print 2>/dev/null \
        | grep -E "/filtered_sim_s${STATION}_${ENERGY}eV_r0*${RUN}\.root$" \
        | head -n 1)


    if [[ -n "${OUTPUT_FOUND}" ]]; then

        # Record this run immediately so that if the task hangs
        # on a later run, this completed run remains tracked.
        echo "${ENERGY} ${RUN}" \
            >> "${PROCESSED_FILE}"


        N_PROCESSED=$((N_PROCESSED + 1))


        {
            echo "completed_energy=${ENERGY}"
            echo "completed_run=${RUN}"
            echo "completed_run_time=$(date --iso-8601=seconds)"
        } >> "${TASK_STATUS_FILE}"


        echo "Run completed successfully."
        echo "Output:"
        echo "${OUTPUT_FOUND}"


    else

        echo "ERROR: Python returned success, but expected output was not found."


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
echo "Input-missing file:      ${INPUT_MISSING_FILE}"
echo "Output-missing file:     ${OUTPUT_MISSING_FILE}"
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
