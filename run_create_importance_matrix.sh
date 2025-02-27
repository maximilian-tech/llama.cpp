#!/bin/env bash

SOURCE_DIR="/data/horse/ws/s0872522-llm-zfp/llama.cpp/"
EXEC_DIR="/home/s0872522/workspaces/cat/s0872522-llm-zfp/llama.cpp"

export NCPUS=8
SETTINGS="-ngl 300 -s 1 -t ${NCPUS} --ctx-size 4096 "


#models=( "3-8B" "3-70B" "3.1-8B" "3.1-70B" )
models=( "3-70B" "3.1-8B" "3.1-70B" )
SOURCE_TYPE="F16"

for model in "${models[@]}" ; do
    MODEL_SOURCEDIR="${SOURCE_DIR}Meta-Llama-${model}"
    MODEL_PREFIX="${MODEL_SOURCEDIR}/Meta-Llama-${model}"
    
    sbatch << EOF
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -A p_darwin
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:2

cd $EXEC_DIR

source ./modules.rc
    
srun ./_build/bin/llama-imatrix \
    -m ${MODEL_PREFIX}-${SOURCE_TYPE}.gguf \
    -f ${SOURCE_DIR}/wiki.train.raw \
    -o ${MODEL_SOURCEDIR}/imatrix.dat \
    ${SETTINGS}

EOF
    
done
