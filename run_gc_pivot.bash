#!/bin/bash

# =====================================================================
#  Capability-paper pivot: controllable deformation at near-parity quality.
#   1. Train a parity-tightened hybrid (2048 nodes, 6000 iters).
#   2. Measure render FPS (graph vs vanilla) + parameter counts.
#   3. Render the node-editing demo (the unique capability).
#  Submit with:  sbatch run_gc_pivot.bash
# =====================================================================

#SBATCH --job-name=gc_pivot
#SBATCH --time=00:50:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_pivot_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

SCENE=pulling
DATA=data/endonerf/$SCENE
CFG=arguments/endonerf/pulling_graph_hybrid_2048.py
EXP=endonerf/pulling_hybrid2048

echo "=================== JOB START ==================="; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. parity-tightening train
python3 train.py -s "$DATA" --port 6401 --expname "$EXP" --configs "$CFG" --save_iterations 1000 6000

# 2. quality + FPS (render WITH video => render.py prints FPS); also FPS for vanilla baseline
echo "---- render + FPS [hybrid2048] ----"
python3 render.py --model_path output/"$EXP" --configs "$CFG" --skip_train --iteration 6000
python3 metrics.py --model_path output/"$EXP"
echo "---- render + FPS [vanilla baseline] ----"
python3 render.py --model_path output/endonerf/pulling --configs arguments/endonerf/pulling.py --skip_train

# parameter counts (deformation field: graph vs the grid/MLP it sits on)
echo "---- PARAM COUNTS ----"
python3 - <<'PY'
import torch
from argparse import ArgumentParser
from arguments import ModelHiddenParams
from utils.config_loader import load_config
from utils.params_utils import merge_hparams
from scene import GaussianModel
def build(cfg):
    p=ArgumentParser(); hp=ModelHiddenParams(p); a=hp.extract(merge_hparams(p.parse_args([]), load_config(cfg)))
    return GaussianModel(3,a)
for name,cfg in [("vanilla","arguments/endonerf/pulling.py"),("hybrid2048","arguments/endonerf/pulling_graph_hybrid_2048.py")]:
    g=build(cfg); d=g._deformation
    grid=sum(p.numel() for p in d.get_grid_parameters())
    mlp=sum(p.numel() for p in d.get_mlp_parameters())
    node=sum(p.numel() for p in d.get_node_parameters())
    print(f"{name:10s} grid={grid:>8d} mlp={mlp:>6d} node_gnn={node:>6d} total_deform={grid+mlp+node:>8d}")
PY

# 3. editing capability demo
echo "---- EDIT DEMO [hybrid2048] ----"
python3 edit_demo.py --model_path output/"$EXP" --configs "$CFG" --iteration 6000 --edit_mag 0.25 --axis y

echo "==================== JOB END ===================="; date
deactivate
