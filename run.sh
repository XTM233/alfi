#!/bin/zsh

# cd ~/Documents/Github/alfi
# conda activate torch_gp_transcription_dynamics
conda activate local_alfi

for SEED in 42 44; do
    echo "Running experiments with SEED: $SEED"
    NEXT_SEED=$((SEED + 1))

    python pde_experiment.py --gene kr --SEED "$SEED" --max_epoch 300 --note "final-no-zero-mean-sample-10-lr-0.005" --trans "" &

    python pde_experiment.py --gene kr --SEED "$NEXT_SEED" --max_epoch 300 --note "final-no-zero-mean-sample-10-lr-0.005" --trans "" &
    # Wait for both background jobs to finish before moving to the next seed
    wait
done

