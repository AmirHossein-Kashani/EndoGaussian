#!/bin/bash

# =====================================================================
#  Evaluation breadth: Hamlyn seq1 (ForPlane processing) — a THIRD dataset
#  beyond EndoNeRF and SuPer. Trains the four residual-story configs
#  (vanilla / GC-match / SC-GS / SC-GS+residual), renders test, metrics.
#  Submit with:  sbatch run_hamlyn.bash
# =====================================================================

#SBATCH --job-name=hamlyn
#SBATCH --time=03:00:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_hamlyn_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

echo "=================== JOB START ==================="; date

PORT=6900
run () {  # tag config
    local exp="hamlyn/seq1_$1"; local cfg=$2
    PORT=$((PORT+1))
    echo ""; echo "######## hamlyn seq1 $1 ($cfg) port=$PORT ########"
    python3 train.py -s data/hamlyn/hamlyn_seq1 --port "$PORT" --expname "$exp" \
        --configs "$cfg" --save_iterations 3000
    python3 render.py --model_path "output/$exp" --configs "$cfg" \
        --iteration 3000 --skip_train --skip_video
    echo "---- METRICS [seq1 $1] ----"
    python3 metrics.py --model_path "output/$exp"
}

run vanilla     arguments/hamlyn/seq1.py
run match       arguments/hamlyn/seq1_graph_match.py
run scgs        arguments/hamlyn/seq1_graph_scgs.py
run scgs_hybrid arguments/hamlyn/seq1_graph_scgs_hybrid.py

echo "==================== JOB END ===================="; date
deactivate
