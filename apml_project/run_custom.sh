#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --output=out_custom.log
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/shais/dsgissin/apml_snake/bin/activate.csh
module load tensorflow

python3 Snake.py -P "Custom(); Avoid(epsilon=0);Avoid(epsilon=0)" -D 5000 -s 1000 -l "custom.log" -r 0 -plt 0.01 -pat 0.05 -pit 60
    