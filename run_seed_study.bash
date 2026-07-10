#!/bin/bash

# =====================================================================
#  Seed-stability study (paper gap: single-seed headline deltas of
#  0.13-0.5 dB). Retrains the four Table-3 configs on pulling at the
#  standard 1000+3000 schedule with a fresh seed, then renders + metrics.
#  Paper numbers use seed 6666 (the train.py default), so combined with
#  the existing runs this gives n=4 seeds per config.
#  Submit one job per seed:
#    sbatch --export=SEED=1234 run_seed_study.bash
#    sbatch --export=SEED=2025 run_seed_study.bash
#    sbatch --export=SEED=3407 run_seed_study.bash
# =====================================================================

#SBATCH --job-name=seed_study
#SBATCH --time=03:00:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_seed_study_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

SEED=${SEED:?set SEED via sbatch --export=SEED=<n>}
echo "=================== JOB START (seed $SEED) ==================="; date

PORT=$((6700 + SEED % 200))
run () {  # tag config
    local exp="endonerf/seed_study/${1}_s${SEED}"; local cfg=$2
    PORT=$((PORT+1))
    echo ""; echo "######## $1 seed=$SEED ($cfg) port=$PORT ########"
    python3 train.py -s data/endonerf/pulling --port "$PORT" --expname "$exp" \
        --configs "$cfg" --seed "$SEED" --save_iterations 3000
    python3 render.py --model_path "output/$exp" --configs "$cfg" \
        --iteration 3000 --skip_train --skip_video
    echo "---- METRICS [$1 seed=$SEED] ----"
    python3 metrics.py --model_path "output/$exp"
}

run vanilla     arguments/endonerf/pulling.py
run match       arguments/endonerf/pulling_graph_match_3k.py
run scgs        arguments/endonerf/pulling_graph_scgs.py
run scgs_hybrid arguments/endonerf/pulling_graph_scgs_hybrid.py

echo "==================== JOB END ===================="; date
deactivate
