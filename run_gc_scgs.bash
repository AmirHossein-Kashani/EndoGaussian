#!/bin/bash
# Retrained SC-GS-style learned baseline on the 4 SuPer trials, to rule out "a learned method just beats
# blind classical interpolation." Per trial: train SC-GS proxy -> tracking RPE -> control-from-tracks.
#SBATCH --job-name=gc_scgs
#SBATCH --time=01:30:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_scgs_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
SCGS=arguments/endonerf/pulling_graph_scgs.py
PORT=6940

echo "=================== SCGS JOB START ==================="; date
for T in 3 4 8 9; do
  DATA=data/endonerf/super_trial$T
  TRACKS=data/super/v2_data/trial_$T/rgb/trial_${T}_l_pts.npy
  echo ""; echo "################## TRIAL $T (SC-GS baseline) ##################"
  PORT=$((PORT+1)); python3 train.py -s "$DATA" --port $PORT --expname endonerf/super_t${T}_scgs --configs $SCGS --save_iterations 1000 3000
  echo "---- tracking [trial $T scgs] ----"
  python3 eval_tracking.py --model_path output/endonerf/super_t${T}_scgs --configs $SCGS --tracks "$TRACKS" --pad_v 16 --iteration 3000
  echo "---- control-from-tracks [trial $T scgs] ----"
  python3 eval_control.py  --model_path output/endonerf/super_t${T}_scgs --configs $SCGS --tracks "$TRACKS" --pad_v 16 --iteration 3000
done
echo "==================== SCGS JOB END ===================="; date
deactivate
