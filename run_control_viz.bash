#!/bin/bash
# Dump per-point control-from-tracks predictions across ALL frames (pick a representative frame offline).
#SBATCH --job-name=ctrl_viz
#SBATCH --time=00:20:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_ctrl_viz_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
CFG=arguments/endonerf/pulling_graph_match_3k.py

for T in 3 8 9; do
  M=super_match; [ $T -ne 3 ] && M=super_t${T}_match
  python3 eval_control.py --model_path output/endonerf/$M \
          --configs $CFG --tracks data/super/v2_data/trial_$T/rgb/trial_${T}_l_pts.npy \
          --pad_v 16 --iteration 3000 --dump_frame -2 --dump_K 8
done
deactivate
