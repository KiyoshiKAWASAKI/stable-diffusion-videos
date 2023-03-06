#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu@@cvrl_rtx6k -l gpu=1
#$ -l h=!qa-a10-005&!qa-rtx6k-044&!qa-a10-006
#$ -e errors/
#$ -N generate_78_pairs

# Required modules
module load conda
conda init bash
source activate stable_diffusion

python pipeline.py