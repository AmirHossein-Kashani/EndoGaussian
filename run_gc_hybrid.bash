#!/bin/bash

# =====================================================================
#  GC-EndoGaussian HYBRID iteration: train + render + metrics.
#  Graph (low-freq motion) + per-Gaussian residual (high-freq detail),
#  as-isometric prior (annealed) instead of rigid ARAP. Same nodes/layers/
#  iters as pulling_graph for a clean A/B vs the pure-replace result.
#  Submit with:  sbatch run_gc_hybrid.bash
# =====================================================================

#SBATCH --job-name=gc_hybrid
#SBATCH --time=00:30:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_hybrid_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

SCENE=pulling
DATA=data/endonerf/$SCENE
CONFIG=arguments/endonerf/pulling_graph_hybrid.py
EXP=endonerf/pulling_hybrid

echo "=================== JOB START ==================="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "config=$CONFIG  exp=$EXP"
echo "================================================="

python3 train.py -s "$DATA" --port 6017 --expname "$EXP" --configs "$CONFIG" \
    --save_iterations 1000 3000
python3 render.py --model_path output/"$EXP" --configs "$CONFIG" --skip_train --reconstruct
echo "---- METRICS [hybrid] ----"
python3 metrics.py --model_path output/"$EXP"

echo "==================== JOB END ===================="
date
deactivate
