#!/bin/bash
#SBATCH --job-name=r_q_r
#SBATCH --cpus-per-task=128
#SBATCH --mem=500G
#SBATCH --time=5:00:00
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=regular


# Clear the module environment
module --force purge

# Create a Python virtual environment
module load Python/3.9.6-GCCcore-11.2.0
python3 -m venv $HOME/.envs/first_env

# Activate the virtual environment
source $HOME/.envs/first_env/bin/activate

# Install required packages from requirements.txt
pip install -r $HOME/research-internship/requirements.txt

python3 /home1/s4915372/research-internship/cluster-scripts/create_OC.py --reactions_index 4 --input /home1/s4915372/research-internship/data/covaxxy_merged_25_days.csv --output /home1/s4915372/research-internship/files/opinion-changes-25_days

# Deactivate virtual environment
deactivate