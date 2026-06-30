#!/bin/bash

# =====================================================================
#  Track A: optical-flow supervision (the quality bet). Flow injects GT
#  motion the photometric loss ignores; it should preferentially correct
#  the graph's over-smooth motion field. Compares vanilla+flow and
#  hybrid+flow (two weights) against the no-flow baselines on pulling.
#  Win condition: hybrid+flow >= vanilla (37.27) on PSNR/LPIPS.
#  Submit with:  sbatch run_gc_flow.bash
# =====================================================================

#SBATCH --job-name=gc_flow
#SBATCH --time=01:00:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_flow_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

SCENE=pulling
DATA=data/endonerf/$SCENE

echo "=================== JOB START ==================="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================================================="

PORT=6300
run_one () {
    local tag=$1
    local config=$2
    local lflow=$3
    local exp="endonerf/${tag}"
    PORT=$((PORT + 1))
    echo ""
    echo "########## ${tag}  config=${config}  lambda_flow=${lflow}  port=${PORT} ##########"
    python3 train.py -s "$DATA" --port "$PORT" --expname "$exp" --configs "$config" \
        --lambda_flow "$lflow" --save_iterations 1000 3000
    python3 render.py --model_path output/"$exp" --configs "$config" --skip_train --skip_video
    echo "---- METRICS [${tag}] ----"
    python3 metrics.py --model_path output/"$exp"
}

run_one pulling_vanilla_flow05 arguments/endonerf/pulling.py              0.5
run_one pulling_hybrid_flow05  arguments/endonerf/pulling_graph_hybrid.py 0.5
run_one pulling_hybrid_flow10  arguments/endonerf/pulling_graph_hybrid.py 1.0

echo "==================== JOB END ===================="
date
deactivate
