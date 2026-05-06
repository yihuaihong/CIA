# Source this file to start the GPU keep-alive watchdog as a background
# process for the current sbatch script. Disable per-job via:
#     export ENABLE_KEEPALIVE=0
#
# Usage:
#     source ${REPO_ROOT}/sbatch/_keepalive_init.sh
if [[ "${ENABLE_KEEPALIVE:-1}" == "1" ]]; then
    python ${REPO_ROOT}/scripts/gpu_keepalive.py &
    _KA_PID=$!
    trap "kill $_KA_PID 2>/dev/null" EXIT
    echo "GPU keepalive PID=$_KA_PID"
fi
