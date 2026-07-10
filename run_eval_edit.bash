#!/bin/bash

# =====================================================================
#  Quantitative edit evaluation (paper gap: "edit quality is evaluated
#  qualitatively"). Runs eval_edit.py on the three 3000-iter budget-matched
#  models from Table 3 plus the 6000-iter match model.
#  Submit with:  sbatch run_eval_edit.bash
# =====================================================================

#SBATCH --job-name=eval_edit
#SBATCH --time=00:45:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_eval_edit_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

echo "=================== JOB START ==================="; date

run () {  # exp config iter
    echo ""; echo "######## eval_edit: $1 @ $3 ########"
    python3 eval_edit.py --model_path "output/endonerf/$1" --configs "$2" --iteration "$3"
}

run pulling_match3k      arguments/endonerf/pulling_graph_match_3k.py    3000
run pulling_scgs_hybrid  arguments/endonerf/pulling_graph_scgs_hybrid.py 3000
run pulling_scgs         arguments/endonerf/pulling_graph_scgs.py        3000
run pulling_match        arguments/endonerf/pulling_graph_match.py       6000

echo "==================== JOB END ===================="; date
deactivate
