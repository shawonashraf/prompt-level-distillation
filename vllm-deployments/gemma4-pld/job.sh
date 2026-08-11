#!/bin/bash
#SBATCH --job-name=pld-gemma4
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/sashraf1/prompt-d/slurm-%j.out

set -uo pipefail

WORK=/scratch-shared/sashraf1/prompt-d
REPO=$WORK/prompt-level-distillation
SIF=$WORK/vllm-openai.sif

export HF_HOME=$WORK/hf
export APPTAINER_CACHEDIR=$WORK/apptainer_cache
mkdir -p "$HF_HOME" "$APPTAINER_CACHEDIR" "$WORK/output"

# WANDB_API_KEY / HF_TOKEN live outside the repo, chmod 600
source "$HOME/.pld_env"

if [ ! -f "$SIF" ]; then
    echo "Pulling vLLM container..."
    apptainer pull "$SIF" docker://vllm/vllm-openai:latest
fi

serve() { # serve <model> <port> <gpu-mem-frac> <logfile>
    apptainer exec --nv \
        --bind /scratch-shared:/scratch-shared \
        --env HF_HOME="$HF_HOME" \
        --env HF_TOKEN="${HF_TOKEN:-}" \
        "$SIF" \
        vllm serve "$1" \
            --port "$2" \
            --gpu-memory-utilization "$3" \
            --max-model-len 65536 \
            --reasoning-parser gemma4 \
            --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
            > "$4" 2>&1 &
    echo $!
}

wait_healthy() { # wait_healthy <port> <name> <timeout-sec>
    local waited=0
    until curl -sf "http://localhost:$1/v1/models" > /dev/null; do
        sleep 15
        waited=$((waited + 15))
        if [ "$waited" -ge "$3" ]; then
            echo "FATAL: $2 not healthy after ${3}s"
            tail -50 "$WORK/$2_vllm.log"
            exit 1
        fi
    done
    echo "$2 healthy on port $1 after ${waited}s"
}

trap 'kill $(jobs -p) 2>/dev/null' EXIT

# start teacher first so its memory profiling sees a free GPU, then the student
TEACHER_PID=$(serve google/gemma-4-26B-A4B-it 8000 0.68 "$WORK/teacher_vllm.log")
wait_healthy 8000 teacher 3600   # first run downloads ~53GB of weights

STUDENT_PID=$(serve google/gemma-4-E2B-it 8001 0.18 "$WORK/student_vllm.log")
wait_healthy 8001 student 1800

nvidia-smi --query-gpu=memory.used,memory.total --format=csv

echo "=== Running pipeline ==="
cd "$REPO"
.venv/bin/python main.py -c configs/snellius.yaml -v
STATUS=$?
echo "=== Pipeline exited with status $STATUS ==="
exit $STATUS
