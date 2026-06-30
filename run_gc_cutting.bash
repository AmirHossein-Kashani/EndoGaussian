#!/bin/bash

# =====================================================================
#  GC-EndoGaussian on EndoNeRF cutting_tissues_twice (2nd, more dynamic
#  dataset). Tests whether the negative pulling result is dataset-specific.
#  Trains vanilla and hybrid, renders test, scores standard metrics.
#  Submit with:  sbatch run_gc_cutting.bash
# =====================================================================

#SBATCH --job-name=gc_cutting
#SBATCH --time=00:45:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_cutting_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

SCENE=cutting
DATA=data/endonerf/$SCENE

echo "=================== JOB START ==================="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "frames: $(ls $DATA/images | wc -l) images, $(ls $DATA/depth | wc -l) depth, $(ls $DATA/masks | wc -l) masks"
echo "================================================="

PORT=6200
run_one () {
    local variant=$1
    local config=$2
    local exp="endonerf/cutting_${variant}"
    PORT=$((PORT + 1))
    echo ""
    echo "########## ${variant} on cutting (config=${config}) port=${PORT} ##########"
    python3 train.py -s "$DATA" --port "$PORT" --expname "$exp" --configs "$config" \
        --save_iterations 1000 3000
    python3 render.py --model_path output/"$exp" --configs "$config" --skip_train --skip_video
    echo "---- METRICS [cutting ${variant}] ----"
    python3 metrics.py --model_path output/"$exp"
}

run_one vanilla arguments/endonerf/cutting.py
run_one hybrid  arguments/endonerf/cutting_graph_hybrid.py

echo "==================== JOB END ===================="
date
deactivate
