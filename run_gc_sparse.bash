#!/bin/bash

# =====================================================================
#  Sparse-view robustness sweep: does the graph's coherence prior degrade
#  LESS than vanilla EndoGaussian as training frames are starved?
#  Trains {vanilla, hybrid} at train_frame_stride {1,2,4} and scores the
#  standard (untouched) test set. Win condition: hybrid degrades less /
#  overtakes vanilla at higher stride.
#  Submit with:  sbatch run_gc_sparse.bash
# =====================================================================

#SBATCH --job-name=gc_sparse
#SBATCH --time=01:30:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_sparse_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

SCENE=pulling
DATA=data/endonerf/$SCENE

echo "=================== JOB START ==================="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================================================="

PORT=6100
run_one () {
    local variant=$1     # vanilla | hybrid
    local config=$2
    local stride=$3
    local exp="endonerf/sparse_${variant}_s${stride}"
    PORT=$((PORT + 1))
    echo ""
    echo "########## ${variant}  stride=${stride}  (config=${config}) port=${PORT} ##########"
    python3 train.py -s "$DATA" --port "$PORT" --expname "$exp" --configs "$config" \
        --train_frame_stride "$stride" --save_iterations 1000 3000
    python3 render.py --model_path output/"$exp" --configs "$config" --skip_train --skip_video
    echo "---- METRICS [${variant} s${stride}] ----"
    python3 metrics.py --model_path output/"$exp"
}

for S in 1 2 4; do
    run_one vanilla arguments/endonerf/pulling.py              "$S"
    run_one hybrid  arguments/endonerf/pulling_graph_hybrid.py "$S"
done

echo "==================== JOB END ===================="
date
deactivate
