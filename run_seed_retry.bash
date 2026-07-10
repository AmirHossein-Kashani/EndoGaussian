#!/bin/bash

# Retry the one crashed cell of the seed study: scgs @ seed 3407
# (CUDA illegal memory access right after node seeding — likely transient).
# If the retry crashes again, fall back to a substitute seed 4242 so the
# scgs config still has n=4 seeds.
# Submit with: sbatch run_seed_retry.bash

#SBATCH --job-name=seed_retry
#SBATCH --time=01:00:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_seed_retry_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

echo "=================== JOB START ==================="; date
CFG=arguments/endonerf/pulling_graph_scgs.py

try_seed () {
    local SEED=$1
    local exp="endonerf/seed_study/scgs_s${SEED}"
    rm -rf "output/$exp"
    echo "######## scgs seed=$SEED retry ########"
    if python3 train.py -s data/endonerf/pulling --port $((6800 + SEED % 100)) \
         --expname "$exp" --configs "$CFG" --seed "$SEED" --save_iterations 3000; then
        python3 render.py --model_path "output/$exp" --configs "$CFG" \
            --iteration 3000 --skip_train --skip_video
        echo "---- METRICS [scgs seed=$SEED] ----"
        python3 metrics.py --model_path "output/$exp"
        return 0
    fi
    return 1
}

try_seed 3407 || { echo "!! seed 3407 crashed again — substituting seed 4242"; try_seed 4242; }

echo "==================== JOB END ===================="; date
deactivate
