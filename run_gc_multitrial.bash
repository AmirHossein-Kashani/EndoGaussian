#!/bin/bash
# Multi-trial SuPer (trials 4, 8, 9) to strengthen the controllability result beyond a single trial.
# Per trial: convert -> train vanilla + graph(match) -> tracking RPE (both) -> control-from-tracks (graph).
#SBATCH --job-name=gc_multi
#SBATCH --time=01:45:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_multi_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
VAN=arguments/endonerf/pulling.py
MATCH=arguments/endonerf/pulling_graph_match_3k.py
PORT=6920

echo "=================== JOB START ==================="; date
for T in 4 8 9; do
  DATA=data/endonerf/super_trial$T
  TRACKS=data/super/v2_data/trial_$T/rgb/trial_${T}_l_pts.npy
  echo ""; echo "################## TRIAL $T ##################"
  python3 tools/super_to_endonerf.py data/super/v2_data/trial_$T "$DATA"
  PORT=$((PORT+1)); python3 train.py -s "$DATA" --port $PORT --expname endonerf/super_t${T}_vanilla --configs $VAN   --save_iterations 1000 3000
  PORT=$((PORT+1)); python3 train.py -s "$DATA" --port $PORT --expname endonerf/super_t${T}_match   --configs $MATCH --save_iterations 1000 3000
  echo "---- tracking [trial $T] ----"
  python3 eval_tracking.py --model_path output/endonerf/super_t${T}_vanilla --configs $VAN   --tracks "$TRACKS" --pad_v 16 --iteration 3000
  python3 eval_tracking.py --model_path output/endonerf/super_t${T}_match   --configs $MATCH --tracks "$TRACKS" --pad_v 16 --iteration 3000
  echo "---- control-from-tracks [trial $T] ----"
  python3 eval_control.py  --model_path output/endonerf/super_t${T}_match   --configs $MATCH --tracks "$TRACKS" --pad_v 16 --iteration 3000
done
echo "==================== JOB END ===================="; date
deactivate
