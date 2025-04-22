#!/bin/bash -x

#SBATCH --mem=128G
#SBATCH --gres=gpu:rtxa6000:8
#SBATCH --time=3-00:00:00


module load Python3/3.11.2
# activate you virtual environment or conda
source ./.venv/bin/activate
# the next 3 lines install the fastmax library
module load gcc
module load cuda
python /fs/nexus-scratch/agerami/litgpt/setup_fastmax.py install
# your commands
litgpt download EleutherAI/pythia-1.4b
litgpt pretrain pythia-1.4b \
   --config train_linear.yaml  # location of the "train_linear.yaml" file

# Note: for d = 64, linear attention will be faster than flash attention for roughly > 16k tokens
