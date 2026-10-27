#!/usr/bin/env bash
# Full study of Section 4. Each step can be run independently; runs that
# already exist are skipped, so the script can be resumed.
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-0-19}
PROFILE=${PROFILE:-paper}

# 4.1 emulator of the level-three solver and reference fronts
python scripts/train_emulator.py --problem laminate
python scripts/train_emulator.py --problem polypropylene
for p in laminate polypropylene; do
  python scripts/reference_front.py --problem $p --emulator results/$p/emulator.npz
done
for p in mf_zdt1 mf_zdt3 mf_dtlz2 curewarp4; do
  python scripts/reference_front.py --problem $p
done

# 4.2 synthetic benchmarks
for p in mf_zdt1 mf_zdt3 mf_dtlz2 curewarp4; do
  python scripts/run_experiment.py --problem $p --methods all --seeds $SEEDS --profile $PROFILE
  python scripts/summarize.py --problem $p
done

# 4.3-4.6 case studies, allocation and ablations
python scripts/run_experiment.py --problem laminate --methods all --seeds $SEEDS --profile $PROFILE \
  --emulator results/laminate/emulator.npz
python scripts/run_experiment.py --problem laminate --methods ablations --seeds $SEEDS --profile $PROFILE \
  --emulator results/laminate/emulator.npz
python scripts/run_experiment.py --problem polypropylene --methods all --seeds $SEEDS --profile $PROFILE \
  --emulator results/polypropylene/emulator.npz

# 4.7 sensitivity grid
python scripts/run_sensitivity.py --problem laminate --seeds $SEEDS --profile $PROFILE \
  --emulator results/laminate/emulator.npz

python scripts/summarize.py --problem laminate
python scripts/summarize.py --problem polypropylene

# 4.8 robust designs verified with the level-three solver
python scripts/robust_decode.py --problem laminate --run results/laminate/PE-MFPBO/seed_00.npz \
  --emulator results/laminate/emulator.npz

python scripts/plot_figures.py
