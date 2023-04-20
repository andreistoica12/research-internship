#!/bin/bash
#SBATCH --job-name=preproc
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
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

python3 /home1/s4915372/research-internship/cluster-scripts/preprocessing.py --input /home1/s4915372/research-internship/data/covaxxy_csv/ --output /home1/s4915372/research-internship/data/

# Deactivate virtual environment
deactivate