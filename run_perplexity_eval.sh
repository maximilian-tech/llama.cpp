#!/bin/env bash

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

SOURCE_TYPE="F16"

SOURCE_DIR="/data/horse/ws/s0872522-llm-zfp/llama.cpp/"
EXEC_DIR="/home/s0872522/workspaces/cat/s0872522-llm-zfp/llama.cpp"

EXECUTABLE_PPL="${EXEC_DIR}/_build/bin/llama-perplexity"
EXECUTABLE_CLI="${EXEC_DIR}/_build/bin/llama-cli"

OUTPUT_SUMMARY="${SOURCE_DIR}/log.metric"

HELLASWAG_NTASK=400

CLI_PROMPT="How much wood would a woodchuck chuck if a woodchuck could chuck wood"

export NCPUS=8
SETTINGS="-ngl 300 -s 1 -t ${NCPUS} --ctx-size 4096 "


if [[ "${1:-}" == "test" ]]; then
    models=( "3-70B" )
else
    #models=( "3-8B" "3-70B" "3.1-8B" "3.1-70B" )
    models=( "3-8B" "3.1-8B" )
fi


for model in "${models[@]}"; do

    MODEL_SOURCEDIR="${SOURCE_DIR}Meta-Llama-${model}"
    MODEL_PREFIX="${MODEL_SOURCEDIR}/Meta-Llama-${model}"
    
    NGPUS=$([[ "$model" =~ 70B ]] && echo "2" || echo "1")
    
    if [[ "${1:-}" == "test" ]]; then
        search_command="find ${MODEL_SOURCEDIR}/weights_F16 -type f -size +14G | head -n 1"
    else
        search_command="find ${MODEL_SOURCEDIR}/weights_F16 -type f -size +14G"
    fi
    echo "Search Command: '${search_command}'"
    
    for GGUF_F16_FILE in $(eval "$search_command") ; do
        OUTPUT_NAME="$(basename -- "$GGUF_F16_FILE" .gguf)"
        
        echo "Found Model: ${GGUF_F16_FILE}"
        
        mkdir -p "${MODEL_SOURCEDIR}/jobs_eval"
        mkdir -p "${MODEL_SOURCEDIR}/logs_eval"
        
        JOB_SCRIPT="${MODEL_SOURCEDIR}/jobs_eval/job_script_${model}_${OUTPUT_NAME}.sh"

        cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c ${NCPUS}
#SBATCH --mem=200G
#SBATCH -A p_darwin
#SBATCH --output="${MODEL_SOURCEDIR}/logs_eval/log.${OUTPUT_NAME}_%j.out"
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:${NGPUS}

cd "${EXEC_DIR}"

module purge
source ./modules.rc

srun "${EXECUTABLE_CLI}" \
     ${SETTINGS} \
     -m "${GGUF_F16_FILE}" \
     --repeat_penalty 1.0 \
     --prompt "${CLI_PROMPT}" \
     --predict 200 \
     2>&1 | tee ${MODEL_SOURCEDIR}/logs_eval/${OUTPUT_NAME}.cli

srun "${EXECUTABLE_PPL}" \
     ${SETTINGS} \
     --hellaswag \
     -f "${SOURCE_DIR}/hellaswag_val_full.txt" \
     --hellaswag-tasks ${HELLASWAG_NTASK} \
     -m "${GGUF_F16_FILE}" \
     2>&1 | tee ${MODEL_SOURCEDIR}/logs_eval/${OUTPUT_NAME}.hellaswag

srun "${EXECUTABLE_PPL}" \
     ${SETTINGS} \
     --perplexity \
     --file "${SCRIPT_DIR}/wiki.train.raw" \
     -m "${GGUF_F16_FILE}" \
     2>&1 | tee ${MODEL_SOURCEDIR}/logs_eval/${OUTPUT_NAME}.ppl


PPL_RESULT=\$(grep 'Final estimate: PPL =' "${MODEL_SOURCEDIR}/logs_eval/${OUTPUT_NAME}.ppl" 2>/dev/null) || PPL_RESULT="N/A"
HSWAG_RESULT=\$(grep -Po "(?<=^${HELLASWAG_NTASK}[[:space:]]).+" "${MODEL_SOURCEDIR}/logs_eval/${OUTPUT_NAME}.hellaswag" 2>/dev/null) || HSWAG_RESULT="N/A"

echo "${OUTPUT_NAME} -- \${PPL_RESULT:-N/A} -- HellaSwag: Score = \${HSWAG_RESULT:-N/A}" >> "${OUTPUT_SUMMARY}"

EOF
        sync
        sleep 0.05
        sbatch "$JOB_SCRIPT"

    done # gguf-file
done # model 