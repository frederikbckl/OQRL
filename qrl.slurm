#!/bin/bash
#SBATCH --job-name=quantum_rl
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=All
#SBATCH --array=0-2 

# Adjust the array range based on the number of seeds you have

# Move into the right project folder
cd ~/Desktop/Programming/BachelorThesis/OQRL

# Activate your virtual environment
source ~/Desktop/Programming/BachelorThesis/OQRL/cip-venv/bin/activate

# Make sure logs folder exists
mkdir -p logs

# Manually set safe TMPDIR (optional)
export TMPDIR=/home/b/bickel/tmp
mkdir -p $TMPDIR

# Create a seed list (manually or load from file)
# SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
SEEDS=(1 2 3)


# Pick the correct seed based on the SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Parameters for your training
ENV_NAME="CartPole-v1"
NUM_EPOCHS=1

# Load Python or your environment here if needed
# module load python/3.9  # optional, if needed
# source ~/your_virtualenv/bin/activate  # optional

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

# Run Python script
python3 src/main.py --env_name $ENV_NAME --num_epochs $NUM_EPOCHS --seed $SEED

# Summarize the logs once all jobs are done
cd src
python3 summarize_logs.py

