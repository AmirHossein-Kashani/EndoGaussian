#!/bin/bash
# GAT ablation: match recipe with attention aggregation (gnn_type='gat') vs the EdgeConv match baseline.
# Reconstruction on pulling; tracking + (decontaminated) control on SuPer trial 3.
#SBATCH --job-name=gc_gat
#SBATCH --time=00:50:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_gat_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
GAT=arguments/endonerf/pulling_graph_gat.py

echo "=== PULLING reconstruction (GAT, 3000 iters) ==="
python3 train.py -s data/endonerf/pulling --port 6581 --expname endonerf/pulling_gat --configs $GAT --save_iterations 3000
python3 render.py --model_path output/endonerf/pulling_gat --configs $GAT --iteration 3000 --skip_train --skip_video
python3 metrics.py --model_path output/endonerf/pulling_gat

echo "=== SUPER trial 3 tracking + control (GAT) ==="
python3 train.py -s data/endonerf/super_trial3 --port 6582 --expname endonerf/super_t3_gat --configs $GAT --save_iterations 1000 3000
python3 eval_tracking.py --model_path output/endonerf/super_t3_gat --configs $GAT --tracks data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy --pad_v 16 --iteration 3000
python3 eval_control.py  --model_path output/endonerf/super_t3_gat --configs $GAT --tracks data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy --pad_v 16 --iteration 3000
echo "=== DONE ==="; date
deactivate
