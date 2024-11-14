#!/bin/bash

#SBATCH --time=6-23:59:59
#SBATCH -A p_darwin
#SBATCH -N 1
##SBATCH --exclusive
#SBATCH -n 1 
#SBATCH -c 20
#SBATCH --mem=200G 
#SBATCH --hint=nomultithread
#SBATCH --distribution=block


source ../source_env.rc
source ./python-llamacpp-venv/bin/activate

set -e

export GG="Meta-Llama-3-8B/Meta-Llama-3-8B"
export I="F32"

export OMP_NUM_THREADS=20
export OMP_PLACES=cores
export OMP_PROC_BIND=close

srun -c 20 --distribution=block --extra-node-info=2:51:1 ./build/bin/llama-imatrix -m "${GG}-${I}.gguf" -f "$(dirname ${GG})/wiki.train.raw" -o "${GG}-imatrix.dat" -t 20
