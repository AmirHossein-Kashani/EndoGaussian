#!/bin/bash

# =====================================================================
#  Temporal edit evaluation (workshop feedback #6): handle fidelity,
#  pixel leakage, and foldover at 5 timestamps across the sequence,
#  for the three budget-matched Table-3 models. Also re-benchmarks
#  render FPS for match3k vs scgs_hybrid on the same GPU (feedback #4).
#  Submit with:  sbatch run_eval_temporal.bash
# =====================================================================

#SBATCH --job-name=edit_temporal
#SBATCH --time=01:00:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_edit_temporal_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

echo "=================== JOB START ==================="; date

TIMES="0.1 0.3 0.5 0.7 0.9"

run () {  # exp config
    echo ""; echo "######## temporal eval_edit: $1 ########"
    python3 eval_edit.py --model_path "output/endonerf/$1" --configs "$2" --iteration 3000 \
        --times $TIMES --out_json "output/endonerf/$1/edit_metrics_temporal.json"
}

run pulling_match3k      arguments/endonerf/pulling_graph_match_3k.py
run pulling_scgs_hybrid  arguments/endonerf/pulling_graph_scgs_hybrid.py
run pulling_scgs         arguments/endonerf/pulling_graph_scgs.py

# ---- FPS benchmark on the same GPU (architecture-choice justification) ----
echo ""; echo "######## FPS: pulling_match3k ########"
python3 render.py --model_path output/endonerf/pulling_match3k \
    --configs arguments/endonerf/pulling_graph_match_3k.py --iteration 3000 \
    --skip_train --skip_test 2>&1 | grep -E "FPS:"
echo "######## FPS: pulling_scgs_hybrid ########"
python3 render.py --model_path output/endonerf/pulling_scgs_hybrid \
    --configs arguments/endonerf/pulling_graph_scgs_hybrid.py --iteration 3000 \
    --skip_train --skip_test 2>&1 | grep -E "FPS:"

echo "==================== JOB END ===================="; date
deactivate
