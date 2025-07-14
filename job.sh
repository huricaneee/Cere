#!/bin/bash
#SBATCH --job-name=fine_tune_transformer    # Your job name
#SBATCH --time=15:00:00                   # 15 minutes
#SBATCH --cpus-per-task=1                 # CPU cores
#SBATCH --mem=8G                          # RAM
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --output=logs/output_%j.txt       # Save output logs to a file

# Load Python module
module load python/3.10
module load rust

cd $SCRATCH/Cere
source ~/yujie-env/bin/activate



# (Optional) Add any runtime dependencies here:
# pip install pandas torch wandb etc.




# Install your package from pyproject.toml
cd scripts
wandb online
# Run your Python script
python run_train.py


