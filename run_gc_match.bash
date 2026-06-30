#!/bin/bash

# =====================================================================
#  MATCH experiment: does the adjusted graph (translation-only, no
#  coherence reg, frozen nodes) close the ~0.5 dB gap to vanilla-6k while
#  keeping the drag-to-edit capability? Compare to:
#    pulling vanilla-6k 37.32 / 0.0509 / 2.646  | cutting vanilla-6k 39.42 / 0.0322 / 1.358
#  Submit with:  sbatch run_gc_match.bash
# =====================================================================

#SBATCH --job-name=gc_match
#SBATCH --time=01:00:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_match_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

echo "=================== JOB START ==================="; date

PORT=6600
run () {  # tag config scene
    local exp="endonerf/$1"; local cfg=$2; local scene=$3
    PORT=$((PORT+1))
    echo ""; echo "######## $1  ($cfg)  port=$PORT ########"
    python3 train.py -s "data/endonerf/$scene" --port "$PORT" --expname "$exp" --configs "$cfg" \
        --save_iterations 1000 6000
    python3 render.py --model_path output/"$exp" --configs "$cfg" --skip_train --skip_video --iteration 6000
    echo "---- METRICS [$1] ----"; python3 metrics.py --model_path output/"$exp"
}

run pulling_match arguments/endonerf/pulling_graph_match.py pulling
run cutting_match arguments/endonerf/cutting_graph_match.py cutting

# confirm editing still works in match mode
echo "---- [edit demo] pulling_match ----"
python3 edit_demo.py --model_path output/endonerf/pulling_match \
    --configs arguments/endonerf/pulling_graph_match.py --iteration 6000 \
    --edit_mag 0.06 --radius_frac 0.10 --axis y --out output/endonerf/pulling_match/edit_gentle

echo "==================== JOB END ===================="; date
deactivate
