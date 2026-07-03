#!/bin/bash
#SBATCH --job-name=gatroute4
#SBATCH --time=00:50:00
#SBATCH --account=def-ester
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --output=output_gatroute4_%j.out
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
CFG=arguments/endonerf/pulling_graph_gat_route.py
PORT=6600
for T in 4 8 9; do
  DATA=data/endonerf/super_trial$T; TRACKS=data/super/v2_data/trial_$T/rgb/trial_${T}_l_pts.npy
  PORT=$((PORT+1)); python3 train.py -s "$DATA" --port $PORT --expname endonerf/super_t${T}_gat_route --configs $CFG --save_iterations 1000 3000
  python3 eval_tracking.py --model_path output/endonerf/super_t${T}_gat_route --configs $CFG --tracks "$TRACKS" --pad_v 16 --iteration 3000
  python3 eval_control.py  --model_path output/endonerf/super_t${T}_gat_route --configs $CFG --tracks "$TRACKS" --pad_v 16 --iteration 3000
done
echo DONE; date; deactivate
