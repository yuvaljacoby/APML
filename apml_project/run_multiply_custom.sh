#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --output=out_muli_custom.log
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/shais/dsgissin/apml_snake/bin/activate.csh
module load tensorflow

python3 Snake.py -P "Custom(); Custom(); Custom(); Custom(); Custom()" -D 5000 -s 1000 -l "multi_custom.log" -r 0 -plt 0.1 -pat 0.02 -pit 60
