#!/bin/bash
# run_tensor_comparison.sh

set -euo pipefail
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

SOURCE_TYPE="F16"

SOURCE_DIR="/data/horse/ws/s0872522-llm-zfp/llama.cpp/"
EXEC_DIR="/data/horse/ws/s0872522-llm-zfp/llama.cpp/"

#EXECUTABLE_PPL="${EXEC_DIR}/_build/bin/llama-perplexity"
#EXECUTABLE_CLI="${EXEC_DIR}/_build/bin/llama-cli"
EXECUTABLE_COMP="${EXEC_DIR}/build/bin/llama-compare-tensors"


export NCPUS=1
SETTINGS="--histogram --per-layer-stats"


if [[ "${1:-}" == "test" ]]; then
    models=( "3.1-8B" )
else
    #models=( "3-8B" "3-70B" "3.1-8B" "3.1-70B" )
    models=( "3.1-8B" "3.1-70B" )
    #models=( "3-8B" "3.1-8B" )
fi


for model in "${models[@]}"; do

    MODEL_SOURCEDIR="${SOURCE_DIR}Meta-Llama-${model}"
    MODEL_PREFIX="${MODEL_SOURCEDIR}/Meta-Llama-${model}"
    
    REFERENCE="${MODEL_SOURCEDIR}/weights_F16/Meta-Llama-${model}-F16_F16_no_imat.gguf"
    
    OUTPUT_SUMMARY="${MODEL_SOURCEDIR}/log.model_performance"
    
    RAM=$([[ "$model" =~ 70B ]] && echo "30G" || echo "20G")
    
    if [[ "${1:-}" == "test" ]]; then
        search_command="find ${MODEL_SOURCEDIR}/weights_F16 -type f -size +14G | head -n 1"
    else
        search_command="find ${MODEL_SOURCEDIR}/weights_F16 -type f -size +14G"
    fi
    echo "Search Command: '${search_command}'"
    
    for GGUF_F16_FILE in $(eval "$search_command") ; do
        OUTPUT_NAME="$(basename -- "$GGUF_F16_FILE" .gguf)"
        
        echo "Found Model: ${GGUF_F16_FILE}"
        
        mkdir -p "${MODEL_SOURCEDIR}"/{jobs,logs,result}_tensor_comparison

        JOB_SCRIPT="${MODEL_SOURCEDIR}/jobs_tensor_comparison/job_script_${model}_${OUTPUT_NAME}.sh"
        
        echo "Create Jobsscript '${JOB_SCRIPT}'"
        
        cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c ${NCPUS}
#SBATCH --mem=${RAM}
#SBATCH -A p_darwin
#SBATCH --output="${MODEL_SOURCEDIR}/logs_tensor_comparison/log.${OUTPUT_NAME}_%j.out"
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread

cd "${EXEC_DIR}"

module purge
#source ./modules.rc
source ../source_env_llvm.rc

srun "${EXECUTABLE_COMP}" \
     ${SETTINGS} \
     --model-a "${REFERENCE}" \
     --model-b "${GGUF_F16_FILE}" \
     2>&1 | tee ${MODEL_SOURCEDIR}/result_tensor_comparison/${OUTPUT_NAME}.out

sync ${MODEL_SOURCEDIR}/result_tensor_comparison/${OUTPUT_NAME}.out

EOF
        sync
        #sleep 0.05
        #sbatch "$JOB_SCRIPT"

    done # gguf-file
done # model 
