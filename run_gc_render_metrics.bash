#!/bin/bash

# =====================================================================
#  GC-EndoGaussian: render + metrics ONLY (models already trained by
#  run_gc_endogaussian.bash). Re-run after fixing render.py's CPU-affinity
#  call so it tolerates compute nodes whose cgroup excludes CPU 0.
#  Submit with:  sbatch run_gc_render_metrics.bash
# =====================================================================

#SBATCH --job-name=gc_render
#SBATCH --time=00:25:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_render_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

echo "=================== JOB START ==================="
date

render_and_score () {
    local tag=$1
    local config=$2
    local exp="endonerf/pulling_${tag}"
    echo ""
    echo "########## RENDER+METRICS: ${tag} ##########"
    python3 render.py --model_path output/"$exp" --configs "$config" --skip_train --reconstruct
    echo "---- METRICS [${tag}] ----"
    python3 metrics.py --model_path output/"$exp"
}

render_and_score graph  arguments/endonerf/pulling_graph.py
render_and_score nognn  arguments/endonerf/pulling_graph_nognn.py

echo "==================== JOB END ===================="
date
deactivate
