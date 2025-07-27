cd ~/Documents/Github/alfi
conda activate torch_gp_transcription_dynamics

for SEED in 42 43 44 45 46; do
    python pde_experiment.py --gene kr --SEED $SEED
    python pde_experiment.py --gene kni --SEED $SEED
    python pde_experiment.py --gene gt --SEED $SEED
done