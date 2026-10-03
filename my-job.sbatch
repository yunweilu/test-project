#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --account=def-blaisale
#SBATCH --partition=gh-blais
#SBATCH --time=0-00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --chdir=/net/nfs-iq/data/yunwei/test-project

# Temporary: /home/yunwei doesn't exist yet, so use the data folder as home
export HOME=/net/nfs-iq/data/yunwei
export PATH=$HOME/.local/bin:$PATH

uv sync --extra usecuda
uv run main.py
