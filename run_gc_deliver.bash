#!/bin/bash

# =====================================================================
#  Capability deliverable (robust ordering): FIRST the guaranteed Track B
#  on the already-trained 1024 hybrid (FPS + node-editing demo), THEN the
#  optional parity-tightening retry (now NaN-robust). Even if parity fails,
#  the deliverable is secured.
#  Submit with:  sbatch run_gc_deliver.bash
# =====================================================================

#SBATCH --job-name=gc_deliver
#SBATCH --time=00:45:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_deliver_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
DATA=data/endonerf/pulling

echo "=================== JOB START ==================="; date

# ---- 1. GUARANTEED: graph render FPS + editing demo on the existing 1024 hybrid ----
EXP1=endonerf/pulling_hybrid
CFG1=arguments/endonerf/pulling_graph_hybrid.py
echo "---- [graph FPS] render video on $EXP1 ----"
python3 render.py --model_path output/"$EXP1" --configs "$CFG1" --skip_train --skip_test
echo "---- [edit demo] $EXP1 ----"
python3 edit_demo.py --model_path output/"$EXP1" --configs "$CFG1" --iteration 3000 --edit_mag 0.25 --axis y

# ---- 2. POLISH: parity-tightened 2048 run (now NaN-robust) ----
EXP2=endonerf/pulling_hybrid2048
CFG2=arguments/endonerf/pulling_graph_hybrid_2048.py
echo "---- [parity] train $EXP2 ----"
python3 train.py -s "$DATA" --port 6402 --expname "$EXP2" --configs "$CFG2" --save_iterations 1000 6000
python3 render.py --model_path output/"$EXP2" --configs "$CFG2" --skip_train --skip_video --iteration 6000
echo "---- METRICS [parity 2048] ----"
python3 metrics.py --model_path output/"$EXP2"
echo "---- [edit demo] $EXP2 ----"
python3 edit_demo.py --model_path output/"$EXP2" --configs "$CFG2" --iteration 6000 --edit_mag 0.25 --axis y

echo "==================== JOB END ===================="; date
deactivate
