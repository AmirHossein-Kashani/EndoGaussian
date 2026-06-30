#!/bin/bash
# SuPer tracking-fidelity: convert trial_3 -> EndoNeRF format, train vanilla + graph (match),
# then reprojection-error of the 32 GT-tracked tissue points (vanilla vs graph head-to-head).
#SBATCH --job-name=gc_super
#SBATCH --time=00:50:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_super_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
DATA=data/endonerf/super_trial3
TRACKS=data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy
VAN_CFG=arguments/endonerf/pulling.py
GR_CFG=arguments/endonerf/pulling_graph_match_3k.py

echo "=================== JOB START ==================="; date

# 1. convert SuPer trial_3 -> EndoNeRF format (cv2/opencv available here)
python3 tools/super_to_endonerf.py data/super/v2_data/trial_3 "$DATA"
ls -d "$DATA"/images "$DATA"/depth "$DATA"/masks "$DATA"/poses_bounds.npy

# 2. train vanilla + graph(match) on identical schedule
python3 train.py -s "$DATA" --port 6901 --expname endonerf/super_vanilla --configs "$VAN_CFG" --save_iterations 1000 3000
python3 train.py -s "$DATA" --port 6902 --expname endonerf/super_match   --configs "$GR_CFG"  --save_iterations 1000 3000

# 3. reprojection-error tracking metric on both (frame0 RPE = projection sanity check)
echo "######## TRACKING FIDELITY ########"
python3 eval_tracking.py --model_path output/endonerf/super_vanilla --configs "$VAN_CFG" --tracks "$TRACKS" --pad_v 16 --iteration 3000
python3 eval_tracking.py --model_path output/endonerf/super_match   --configs "$GR_CFG"  --tracks "$TRACKS" --pad_v 16 --iteration 3000

echo "==================== JOB END ===================="; date
deactivate
