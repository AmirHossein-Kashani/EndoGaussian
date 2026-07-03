#!/bin/bash
#SBATCH --job-name=viz_pure
#SBATCH --time=00:15:00
#SBATCH --account=def-ester
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=h100:1
#SBATCH --output=output_viz_pure_%j.out
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
python3 eval_control.py --model_path output/endonerf/super_match --configs arguments/endonerf/pulling_graph_match_3k.py \
        --tracks data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy --pad_v 16 --iteration 3000 --dump_frame 57 --dump_K 8
deactivate
