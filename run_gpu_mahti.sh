#!/bin/bash
#SBATCH --job-name=attn_benchmark
#SBATCH --account=project_2019091
#SBATCH --partition=gpusmall
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

module purge
module load pytorch/2.4

rm -rf ~/.cache/torch_extensions/

python attention_template_gpu.py
