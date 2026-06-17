#!/bin/bash

# =====================================================================
#  EndoGaussian training job for the Digital Research Alliance (Nibi)
#  Submit with:  sbatch run_endogaussian.bash
#  The SBATCH directives must come before any executable line.
# =====================================================================

#SBATCH --job-name=endogauss
#SBATCH --time=00:30:00        # SHORT first run for queue priority. Bump to e.g. 03:00:00 for a full run.
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

echo "=================== JOB START ==================="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))"
echo "================================================="

# --- training ---------------------------------------------------------
# First, a SHORT smoke test on the synthetic scene (verifies the full
# data -> CUDA rasterizer -> optimizer -> checkpoint path runs on the GPU).
python3 train.py \
    -s data/endonerf/synth \
    --port 6017 \
    --expname endonerf/synth \
    --configs arguments/endonerf/synth_quick.py \
    --save_iterations 200 500   # coarse=200, fine=500 (this repo appends the 30k argparse
                                # default to save_iterations BEFORE merging the config, so a
                                # config with non-default iterations never auto-saves otherwise)

# --- For a REAL run, comment the block above and use the real dataset/config:
# python3 train.py -s data/endonerf/pulling --port 6017 \
#     --expname endonerf/pulling --configs arguments/endonerf/pulling.py

echo "==================== JOB END ===================="
date
deactivate
