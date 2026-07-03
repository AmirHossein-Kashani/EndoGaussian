#!/bin/bash
# Control-through-GAT: does routing the edit through the (attention) GNN + a control-consistency loss
# let learned control beat classical interpolation? First check: pulling (reconstruction survives?) +
# SuPer trial 3 (control). If control improves, expand to all 4 trials.
#SBATCH --job-name=gat_route
#SBATCH --time=00:50:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gat_route_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
CFG=arguments/endonerf/pulling_graph_gat_route.py

echo "=== PULLING reconstruction (control-routed GAT, 3000 iters) ==="
python3 train.py -s data/endonerf/pulling --port 6591 --expname endonerf/pulling_gat_route --configs $CFG --save_iterations 3000
python3 render.py --model_path output/endonerf/pulling_gat_route --configs $CFG --iteration 3000 --skip_train --skip_video
python3 metrics.py --model_path output/endonerf/pulling_gat_route

echo "=== SUPER trial 3 tracking + control (control-routed GAT) ==="
python3 train.py -s data/endonerf/super_trial3 --port 6592 --expname endonerf/super_t3_gat_route --configs $CFG --save_iterations 1000 3000
python3 eval_tracking.py --model_path output/endonerf/super_t3_gat_route --configs $CFG --tracks data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy --pad_v 16 --iteration 3000
python3 eval_control.py  --model_path output/endonerf/super_t3_gat_route --configs $CFG --tracks data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy --pad_v 16 --iteration 3000
echo "=== DONE ==="; date
deactivate
