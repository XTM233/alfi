#!/bin/zsh

cd ~/Documents/Github/alfi
conda activate torch_gp_transcription_dynamics

for SEED in 43 44 45 46 47 48 49 50 51; do
    echo "Running experiments with SEED: $SEED"
    python pde_experiment.py --gene kr --SEED $SEED --max_epoch 200 --debug True --note "smaller-lr-0.015"
done