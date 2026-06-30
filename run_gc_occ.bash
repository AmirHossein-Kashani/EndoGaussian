#!/bin/bash

# =====================================================================
#  Occlusion-holdout stress test (THE gate experiment).
#  Trains {vanilla, hybrid} with a central box held out of the loss for a
#  block of mid-sequence frames, renders the train set, and scores tissue
#  recovery inside that box on the occluded frames. Win condition: hybrid
#  has a HIGHER occluded-box PSNR / SMALLER occlusion gap than vanilla.
#  Submit with:  sbatch run_gc_occ.bash
# =====================================================================

#SBATCH --job-name=gc_occ
#SBATCH --time=00:45:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_occ_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

SCENE=pulling
DATA=data/endonerf/$SCENE

echo "=================== JOB START ==================="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================================================="

run_one () {
    local variant=$1
    local config=$2
    local exp="endonerf/occ_${variant}"
    echo ""
    echo "########## ${variant} + occlusion-holdout (config=${config}) ##########"
    python3 train.py -s "$DATA" --port 6017 --expname "$exp" --configs "$config" \
        --occ_holdout --save_iterations 1000 3000
    # render train (for occluded-region eval) + test (standard metrics); skip the slow video
    python3 render.py --model_path output/"$exp" --configs "$config" --skip_video
    echo "---- STANDARD METRICS (test) [${variant}] ----"
    python3 metrics.py --model_path output/"$exp"
    echo "---- OCCLUSION METRICS (train box) [${variant}] ----"
    python3 eval_occlusion.py output/"$exp" 3000
}

run_one vanilla arguments/endonerf/pulling.py
run_one hybrid  arguments/endonerf/pulling_graph_hybrid.py

echo "==================== JOB END ===================="
date
deactivate
