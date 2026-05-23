#!/bin/bash
#SBATCH --job-name=attn_kernel
#SBATCH --account=project_2019091
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8G
#SBATCH --output=attn_kernel_%j.out
#SBATCH --error=attn_kernel_%j.err

module load pytorch/2.4
module load gcc/13.1.0

export LD_PRELOAD=/appl/spack/v020/install-tree/gcc-8.5.0/gcc-13.1.0-how4ki/lib64/libstdc++.so.6

rm -rf ~/.cache/torch_extensions/

srun /appl/soft/ai/wrap/pytorch-2.4/bin/python3 attention_template_gpu.py