#!/bin/bash
# Three legs to address the weaknesses (all on SuPer trial_3 + existing data):
#  Leg 1: tracking with bootstrap CIs + paired Wilcoxon (vanilla vs graph).
#  Leg 2: control-from-tracks controllability metric (graph vs rigid/nearest/TPS, + gnn_layers=0 ablation).
#  Leg 3: stability — vanilla HexPlane diverges (grad_clip=0) while the graph stays finite.
#SBATCH --job-name=gc_improve
#SBATCH --time=01:30:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_gc_improve_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

DATA=data/endonerf/super_trial3
TRACKS=data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy
VAN=arguments/endonerf/pulling.py
MATCH=arguments/endonerf/pulling_graph_match_3k.py
NOGNN=arguments/endonerf/pulling_graph_match_3k_nognn.py
PORT=6910

echo "=================== JOB START ==================="; date
# ensure converted data exists
[ -f "$DATA/poses_bounds.npy" ] || python3 tools/super_to_endonerf.py data/super/v2_data/trial_3 "$DATA"

# --- base models (self-contained) ---
PORT=$((PORT+1)); python3 train.py -s $DATA --port $PORT --expname endonerf/super_vanilla --configs $VAN   --save_iterations 1000 3000
PORT=$((PORT+1)); python3 train.py -s $DATA --port $PORT --expname endonerf/super_match   --configs $MATCH --save_iterations 1000 3000

echo ""; echo "######## LEG 1: tracking fidelity + paired stats ########"
python3 eval_tracking.py --model_path output/endonerf/super_vanilla --configs $VAN   --tracks $TRACKS --pad_v 16 --iteration 3000
python3 eval_tracking.py --model_path output/endonerf/super_match   --configs $MATCH --tracks $TRACKS --pad_v 16 --iteration 3000
python3 eval_paired.py output/endonerf/super_vanilla output/endonerf/super_match

echo ""; echo "######## LEG 2: control-from-tracks controllability ########"
python3 eval_control.py --model_path output/endonerf/super_match --configs $MATCH --tracks $TRACKS --pad_v 16 --iteration 3000
PORT=$((PORT+1)); python3 train.py -s $DATA --port $PORT --expname endonerf/super_match_nognn --configs $NOGNN --save_iterations 1000 3000
echo "---- ablation: gnn_layers=0 ----"
python3 eval_control.py --model_path output/endonerf/super_match_nognn --configs $NOGNN --tracks $TRACKS --pad_v 16 --iteration 3000

echo ""; echo "######## LEG 3: stability (grad_clip=0, 2 seeds) ########"
for S in 6666 1234; do
  PORT=$((PORT+1)); python3 train.py -s $DATA --port $PORT --expname endonerf/super_van_nc_$S   --configs $VAN   --grad_clip 0 --seed $S --save_iterations 3000 || echo "van_nc_$S crashed"
  PORT=$((PORT+1)); python3 train.py -s $DATA --port $PORT --expname endonerf/super_match_nc_$S --configs $MATCH --grad_clip 0 --seed $S --save_iterations 3000 || echo "match_nc_$S crashed"
done
echo "---- stability check (NaN = diverged) ----"
for e in super_van_nc_6666 super_van_nc_1234 super_match_nc_6666 super_match_nc_1234; do
  python3 - "$e" <<'PY'
import sys, glob, numpy as np
from plyfile import PlyData
e=sys.argv[1]; fs=glob.glob(f'output/endonerf/{e}/point_cloud/iteration_3000/point_cloud.ply')
if not fs: print(f'{e}: NO MODEL (diverged/failed)'); raise SystemExit
d=PlyData.read(fs[0]); x=np.stack([d['vertex']['x'],d['vertex']['y'],d['vertex']['z']],1)
print(f'{e}: {"NaN (DIVERGED)" if np.isnan(x).any() else "finite (STABLE)"}')
PY
done

echo "==================== JOB END ===================="; date
deactivate
