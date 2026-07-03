#!/bin/bash
# Residual-matched SC-GS ablation (reviewer request): does giving the SC-GS-style baseline our per-Gaussian
# residual close the reconstruction/tracking gap? Isolates the residual's contribution from the full recipe.
#SBATCH --job-name=scgs_resid
#SBATCH --time=01:00:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_scgs_resid_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
SCGS=arguments/endonerf/pulling_graph_scgs.py
SCGSH=arguments/endonerf/pulling_graph_scgs_hybrid.py
MATCH=arguments/endonerf/pulling_graph_match_3k.py

echo "=== PULLING reconstruction: SC-GS(no resid) vs SC-GS+residual vs match, all 3000 iters ==="
python3 train.py -s data/endonerf/pulling --port 6571 --expname endonerf/pulling_scgs        --configs $SCGS  --save_iterations 3000
python3 render.py --model_path output/endonerf/pulling_scgs        --configs $SCGS  --iteration 3000 --skip_train --skip_video
python3 metrics.py --model_path output/endonerf/pulling_scgs

python3 train.py -s data/endonerf/pulling --port 6572 --expname endonerf/pulling_scgs_hybrid --configs $SCGSH --save_iterations 3000
python3 render.py --model_path output/endonerf/pulling_scgs_hybrid --configs $SCGSH --iteration 3000 --skip_train --skip_video
python3 metrics.py --model_path output/endonerf/pulling_scgs_hybrid

# match reference (already trained at 3000); (re)render test + metrics for an apples-to-apples 3000-iter PSNR
python3 render.py --model_path output/endonerf/pulling_match3k     --configs $MATCH --iteration 3000 --skip_train --skip_video
python3 metrics.py --model_path output/endonerf/pulling_match3k

echo "=== SUPER trial 3 tracking/control: SC-GS+residual ==="
python3 train.py -s data/endonerf/super_trial3 --port 6577 --expname endonerf/super_t3_scgs_hybrid --configs $SCGSH --save_iterations 1000 3000
python3 eval_tracking.py --model_path output/endonerf/super_t3_scgs_hybrid --configs $SCGSH --tracks data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy --pad_v 16 --iteration 3000
python3 eval_control.py  --model_path output/endonerf/super_t3_scgs_hybrid --configs $SCGSH --tracks data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy --pad_v 16 --iteration 3000
echo "=== DONE ==="; date
deactivate
