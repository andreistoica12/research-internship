#!/bin/bash
#SBATCH --job-name=repl_ret
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=4:00:00
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

python3 /home1/s4915372/research-internship/cluster-scripts/create-OC/create_OC_replied_to_retweeted.py --input /home1/s4915372/research-internship/data/covaxxy_merged_25_days.csv --output /home1/s4915372/research-internship/files/opinion-changes-25_days/

# Deactivate virtual environment
deactivate