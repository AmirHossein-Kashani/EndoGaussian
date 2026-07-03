#!/bin/bash
# DIAGNOSTIC: re-run control-from-tracks with the pure-control guard (hybrid residual frozen under
# control_only), so every model is scored on control ALONE and the comparison is fair vs residual-free
# baselines (SC-GS, gnn=0). Overwrites control_results.json; residual-active copies saved as *_residual.json.
#SBATCH --job-name=ctrl_fair
#SBATCH --time=00:40:00
#SBATCH --account=def-ester    # DO NOT MODIFY THIS LINE
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mail-user=amirhossein_kashani@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=output_ctrl_fair_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
MATCH=arguments/endonerf/pulling_graph_match_3k.py
NOGNN=arguments/endonerf/pulling_graph_match_3k_nognn.py
SCGS=arguments/endonerf/pulling_graph_scgs.py

echo "=== FAIR CONTROL EVAL START ==="; date
for T in 3 4 8 9; do
  TRACKS=data/super/v2_data/trial_$T/rgb/trial_${T}_l_pts.npy
  MM=super_match; [ $T -ne 3 ] && MM=super_t${T}_match
  echo "#### trial $T: match (pure control) ####"
  python3 eval_control.py --model_path output/endonerf/$MM --configs $MATCH --tracks "$TRACKS" --pad_v 16 --iteration 3000
  echo "#### trial $T: scgs (pure control, unchanged) ####"
  python3 eval_control.py --model_path output/endonerf/super_t${T}_scgs --configs $SCGS --tracks "$TRACKS" --pad_v 16 --iteration 3000
done
echo "#### trial 3: nognn ablation (pure control) ####"
python3 eval_control.py --model_path output/endonerf/super_match_nognn --configs $NOGNN --tracks data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy --pad_v 16 --iteration 3000
echo "=== FAIR CONTROL EVAL END ==="; date
deactivate
