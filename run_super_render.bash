#!/bin/bash
# Render the full-sequence reconstruction video for the SuPer control-graph models (trials 3,4,8,9).
#SBATCH --job-name=super_render
#SBATCH --time=00:30:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_super_render_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
CFG=arguments/endonerf/pulling_graph_match_3k.py

echo "=== SUPER RENDER START ==="; date
for M in super_match super_t4_match super_t8_match super_t9_match; do
  echo "#### rendering $M ####"
  python3 render.py --model_path output/endonerf/$M --configs $CFG \
          --iteration 3000 --skip_train --skip_test
done
echo "=== SUPER RENDER END ==="; date
deactivate
