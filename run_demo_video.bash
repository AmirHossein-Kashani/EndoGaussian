#!/bin/bash

# Retraction-reveal demo video (workshop supplementary).
# Submit with: sbatch run_demo_video.bash

#SBATCH --job-name=demo_video
#SBATCH --time=00:30:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_demo_video_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

echo "=================== JOB START ==================="; date

python3 tools/make_demo_video.py \
    --model_path output/endonerf/cutting_match \
    --configs arguments/endonerf/cutting_graph_match.py --iteration 6000 \
    --out docs/supplementary/demo_reveal.mp4

echo "==================== JOB END ===================="; date
deactivate
