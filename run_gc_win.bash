#!/bin/bash

# =====================================================================
#  Strengthen the capability paper:
#   1. Clean editing-demo figures (gentle magnitude) on existing models.
#   2. Iteration-matched (6000) capability comparison on BOTH datasets:
#      vanilla vs 2048-node hybrid, pulling + cutting.
#  Submit with:  sbatch run_gc_win.bash
# =====================================================================

#SBATCH --job-name=gc_win
#SBATCH --time=01:30:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_win_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

echo "=================== JOB START ==================="; date

# ---- 1. clean editing-demo figures (gentle, local) on existing models ----
echo "######## clean edit demos ########"
python3 edit_demo.py --model_path output/endonerf/pulling_hybrid2048 \
    --configs arguments/endonerf/pulling_graph_hybrid_2048.py --iteration 6000 \
    --edit_mag 0.06 --radius_frac 0.10 --axis y --out output/endonerf/pulling_hybrid2048/edit_gentle
python3 edit_demo.py --model_path output/endonerf/pulling_hybrid \
    --configs arguments/endonerf/pulling_graph_hybrid.py --iteration 3000 \
    --edit_mag 0.06 --radius_frac 0.10 --axis y --out output/endonerf/pulling_hybrid/edit_gentle

PORT=6500
run () {  # tag config scene iter
    local exp="endonerf/$1"; local cfg=$2; local scene=$3; local it=$4
    PORT=$((PORT+1))
    echo ""; echo "######## $1  ($cfg)  port=$PORT ########"
    python3 train.py -s "data/endonerf/$scene" --port "$PORT" --expname "$exp" --configs "$cfg" \
        --save_iterations 1000 "$it"
    python3 render.py --model_path output/"$exp" --configs "$cfg" --skip_train --skip_video --iteration "$it"
    echo "---- METRICS [$1] ----"; python3 metrics.py --model_path output/"$exp"
}

# ---- 2. iteration-matched capability comparison (6000 iters) ----
#   pulling hybrid-2048@6000 already trained (output/endonerf/pulling_hybrid2048).
run pulling_vanilla6k  arguments/endonerf/pulling_v6000.py             pulling 6000
run cutting_vanilla6k  arguments/endonerf/cutting_v6000.py             cutting 6000
run cutting_hybrid2048 arguments/endonerf/cutting_graph_hybrid_2048.py cutting 6000

echo "---- [edit demo] cutting hybrid ----"
python3 edit_demo.py --model_path output/endonerf/cutting_hybrid2048 \
    --configs arguments/endonerf/cutting_graph_hybrid_2048.py --iteration 6000 \
    --edit_mag 0.06 --radius_frac 0.10 --axis y --out output/endonerf/cutting_hybrid2048/edit_gentle

echo "==================== JOB END ===================="; date
deactivate
