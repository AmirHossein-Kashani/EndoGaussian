#!/bin/bash

# =====================================================================
#  Cut-aware graph experiment on cutting_tissues_twice.
#  Tests the one idea that exploits what HexPlane CANNOT do: represent a
#  discontinuity. Breakable node edges should sharpen the cut boundary.
#  Compares cut-region PSNR across:
#     vanilla-6k  |  graph match (no cut)  |  graph cut-aware
#  Submit with:  sbatch run_gc_cut.bash
# =====================================================================

#SBATCH --job-name=gc_cut
#SBATCH --time=00:50:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_cut_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
DATA=data/endonerf/cutting

echo "=================== JOB START ==================="; date

# ---- 1. train the cut-aware model ----
echo "######## train cutting_cut (cut-aware) ########"
python3 train.py -s "$DATA" --port 6701 --expname endonerf/cutting_cut \
    --configs arguments/endonerf/cutting_graph_cut.py --save_iterations 1000 6000
python3 render.py --model_path output/endonerf/cutting_cut \
    --configs arguments/endonerf/cutting_graph_cut.py --skip_video --iteration 6000
echo "---- standard METRICS [cutting_cut] ----"; python3 metrics.py --model_path output/endonerf/cutting_cut

# ---- 2. render TRAIN sets for the two existing baselines (needed for the cut-region metric) ----
python3 render.py --model_path output/endonerf/cutting_vanilla6k \
    --configs arguments/endonerf/cutting_v6000.py --skip_test --skip_video --iteration 6000
python3 render.py --model_path output/endonerf/cutting_match \
    --configs arguments/endonerf/cutting_graph_match.py --skip_test --skip_video --iteration 6000

# ---- 3. cut-region metric across all three ----
echo "######## CUT-REGION COMPARISON ########"
python3 eval_cut.py output/endonerf/cutting_vanilla6k 6000
python3 eval_cut.py output/endonerf/cutting_match     6000
python3 eval_cut.py output/endonerf/cutting_cut       6000

echo "==================== JOB END ===================="; date
deactivate
