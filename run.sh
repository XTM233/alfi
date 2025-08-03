#!/bin/zsh

cd ~/Documents/Github/alfi
conda activate torch_gp_transcription_dynamics

for SEED in 47 48 49 50 51; do
    echo "Running experiments with SEED: $SEED"
    python pde_experiment.py --gene kr --SEED $SEED 
    python pde_experiment.py --gene kni --SEED $SEED
    python pde_experiment.py --gene gt --SEED $SEED --max_epoch 150
done