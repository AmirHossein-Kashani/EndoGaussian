#!/bin/bash
# Verify the capability+match holds at the ORIGINAL 3000-iter budget (no extra training time),
# and measure FPS. Compare to vanilla@3000 (output/endonerf/pulling = 37.27 / 0.0609 / 2.906).
#SBATCH --job-name=gc_m3k
#SBATCH --time=00:30:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_m3k_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
EXP=endonerf/pulling_match3k
CFG=arguments/endonerf/pulling_graph_match_3k.py

echo "=================== JOB START ==================="; date
python3 train.py -s data/endonerf/pulling --port 6801 --expname "$EXP" --configs "$CFG" --save_iterations 1000 3000
python3 render.py --model_path output/"$EXP" --configs "$CFG" --skip_train --iteration 3000   # video -> FPS
echo "---- METRICS [match@3000] ----"; python3 metrics.py --model_path output/"$EXP"
echo "==================== JOB END ===================="; date
deactivate
