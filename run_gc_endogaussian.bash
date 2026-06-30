#!/bin/bash

# =====================================================================
#  GC-EndoGaussian gate experiment on the Digital Research Alliance (Nibi)
#  Submit with:  sbatch run_gc_endogaussian.bash
#
#  Trains TWO models on the real EndoNeRF pulling clip and renders+scores both:
#    (A) pulling_graph        -> full control-node GNN (gnn_layers=2)
#    (B) pulling_graph_nognn  -> SC-GS-style ablation  (gnn_layers=0)
#  Compare A vs B to decide the gate: does message passing actually help?
#  (Occlusion-holdout stress test comes next; this first run validates the
#   end-to-end CUDA path and gives the standard-setting A/B numbers.)
# =====================================================================

#SBATCH --job-name=gc_endogauss
#SBATCH --time=01:00:00        # two short runs (1000 coarse + 3000 fine each) + render + metrics
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_endogauss_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

SCENE=pulling
DATA=data/endonerf/$SCENE

echo "=================== JOB START ==================="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))"
ls -d "$DATA"/images "$DATA"/depth "$DATA"/masks "$DATA"/poses_bounds.npy 2>&1
echo "================================================="

run_variant () {
    local tag=$1            # graph | nognn
    local config=$2
    local exp="endonerf/pulling_${tag}"
    echo ""
    echo "########## VARIANT: ${tag}  (config=${config}) ##########"
    python3 train.py -s "$DATA" --port 6017 --expname "$exp" --configs "$config" \
        --save_iterations 1000 3000
    python3 render.py --model_path output/"$exp" --configs "$config" --skip_train --reconstruct
    echo "---- METRICS [${tag}] ----"
    python3 metrics.py --model_path output/"$exp"
}

# (A) full control-node GNN
run_variant graph  arguments/endonerf/pulling_graph.py
# (B) SC-GS-style ablation (no message passing)
run_variant nognn  arguments/endonerf/pulling_graph_nognn.py

echo "==================== JOB END ===================="
date
deactivate
