#!/bin/bash

# =====================================================================
#  EndoGaussian REAL run on the Digital Research Alliance (Nibi)
#  Submit with:  sbatch run_endogaussian.bash
#  The SBATCH directives must come before any executable line.
#
#  This trains on a real EndoNeRF clip (default: pulling_soft_tissues),
#  then renders the held-out test set + the time-interpolated video and
#  exports per-frame point clouds, then computes PSNR/SSIM/LPIPS/RMSE.
#  A short synthetic smoke test is kept at the bottom (commented out).
# =====================================================================

#SBATCH --job-name=endogauss
#SBATCH --time=00:45:00        # real pulling run (1000 coarse + 3000 fine) trains in minutes on an H100;
                               # this is mostly headroom for render + metrics + LPIPS setup.
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G              # total CPU RAM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8      # dataloader / CPU side
#SBATCH --gpus=h100:1          # one H100 (80G). Needs StdEnv/2023-era Python (3.8 wheels do NOT support H100).
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_endogauss_%j.out

# --- environment (compute nodes have NO internet; everything is preinstalled in .venv) ---
# opencv/4.11.0 provides cv2 (the pip opencv-python wheel is a dummy on Alliance).
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate

# H100 compute capability, so any lazy CUDA-extension JIT targets the right arch
export TORCH_CUDA_ARCH_LIST="9.0"

# --- scene selection (switch to 'cutting' to use cutting_tissues_twice) ---
SCENE=pulling
DATA=data/endonerf/$SCENE
CONFIG=arguments/endonerf/$SCENE.py
EXP=endonerf/$SCENE

echo "=================== JOB START ==================="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))"
echo "scene=$SCENE  data=$DATA  config=$CONFIG"
ls -d "$DATA"/images "$DATA"/depth "$DATA"/masks "$DATA"/poses_bounds.npy 2>&1
echo "================================================="

# --- 1. Train: two-stage coarse(1000) -> fine(3000) on the real clip ------
#   --save_iterations 1000 3000 forces a checkpoint at the end of BOTH stages.
#   (train.py appends the 30k argparse default to --save_iterations before the
#    config merge, so we name the real stage boundaries explicitly to be safe.)
python3 train.py \
    -s "$DATA" \
    --port 6017 \
    --expname "$EXP" \
    --configs "$CONFIG" \
    --save_iterations 1000 3000

# --- 2. Render: held-out test set + time-interpolated video, plus 3D export -
#   (skip the train set to save time; test is needed for metrics, video is the
#    headline result, --reconstruct dumps fused RGBD point clouds per frame.)
python3 render.py \
    --model_path output/"$EXP" \
    --configs "$CONFIG" \
    --skip_train \
    --reconstruct

# --- 3. Evaluate: PSNR / SSIM / LPIPS / RMSE on the rendered test set ------
python3 metrics.py --model_path output/"$EXP"

echo "==================== JOB END ===================="
date
deactivate

# =====================================================================
#  Synthetic smoke test (pipeline check only — NOT real data). Uncomment
#  to verify the data -> CUDA rasterizer -> optimizer -> checkpoint path.
# =====================================================================
# python3 train.py -s data/endonerf/synth --port 6017 \
#     --expname endonerf/synth --configs arguments/endonerf/synth_quick.py \
#     --save_iterations 200 500
