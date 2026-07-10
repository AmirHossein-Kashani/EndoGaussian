#!/bin/bash

# Representative demo video (workshop supplementary): multi-scene surgery replays,
# residual-ablation-over-time, and a brief paper-figure edit.
# Submit with: sbatch run_demo_video.bash

#SBATCH --job-name=demo_video
#SBATCH --time=00:50:00
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

# missing video splits used by segments 1-2
if [ ! -d output/endonerf/pulling_scgs/video/ours_3000/renders ]; then
  echo "######## render pulling_scgs video split ########"
  python3 render.py --model_path output/endonerf/pulling_scgs \
      --configs arguments/endonerf/pulling_graph_scgs.py --iteration 3000 \
      --skip_train --skip_test
fi
if [ ! -d output/endonerf/cutting_match/video/ours_6000/renders ]; then
  echo "######## render cutting_match video split ########"
  python3 render.py --model_path output/endonerf/cutting_match \
      --configs arguments/endonerf/cutting_graph_match.py --iteration 6000 \
      --skip_train --skip_test
fi

echo "######## compose demo video ########"
python3 tools/make_demo_video.py \
    --model_path output/endonerf/pulling_match \
    --configs arguments/endonerf/pulling_graph_match.py --iteration 6000 \
    --out docs/supplementary/demo.mp4

echo "==================== JOB END ===================="; date
deactivate
