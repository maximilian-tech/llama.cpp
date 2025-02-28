#!/bin/env bash
: '
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=70G
#SBATCH -A p_darwin
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1


SOURCE_DIR="/data/horse/ws/s0872522-llm-zfp/llama.cpp/"
PREFIX="${SOURCE_DIR}Meta-Llama-3-8B/Meta-Llama-3-8B"
EXEC_DIR="/home/s0872522/workspaces/cat/s0872522-llm-zfp/llama.cpp"

cd $EXEC_DIR

source ./modules.rc

OUTPUT_SUMMARY="${SOURCE_DIR}Meta-Llama-3-8B/metric.summary_zfp_quant"

echo "" > $OUTPUT_SUMMARY

#PREFIX="./Meta-Llama-3-8B/Meta-Llama-3-8B"

DIM="4"
export NCPUS=8
HELLASWAG_NTASK=1000
SETTINGS="-ngl 300 -s 1 -t ${NCPUS} --ctx-size 4096 "

# for model in F16 Q8_0 Q4_0 ; do

#     ./_build/bin/llama-perplexity ${SETTINGS} -m ${PREFIX}-${model}.gguf  2>&1 | tee ppl.${model}
#     echo "${model} -- $(grep 'Final estimate: PPL =' ppl.${model})" >> "$OUTPUT_SUMMARY"

# done
for DIM in 4 3 2 1 ; do
    for rate in 3.50 4.00 4.50 4.65 5.00 6.00 8.00 ; do
        echo $rate
        export ZFP_RATE=$rate

        #OUTPUT_NAME="from_ZFP-RATE_${ZFP_RATE}_dim_${DIM}"
        
        #OUTPUT_NAME="from_ZFP-RATE_4.00-6.00_dim_3_new"
        OUTPUT_NAME="from_ZFP-RATE_4.10-4.10_dim_3_new"
        srun ./_build/bin/llama-perplexity ${SETTINGS} --perplexity --file ./Meta-Llama-3-8B/wiki.train.raw -m "${PREFIX}-F16_${OUTPUT_NAME}.gguf" 2>&1 | tee ppl.${OUTPUT_NAME}
	    srun ./_build/bin/llama-perplexity ${SETTINGS} --hellaswag -f "${SOURCE_DIR}/hellaswag_val_full.txt" --hellaswag-tasks ${HELLASWAG_NTASK} -m "${PREFIX}-F16_${OUTPUT_NAME}.gguf" 2>&1 | tee hellaswag.${OUTPUT_NAME}
	    echo "${OUTPUT_NAME} -- $(grep 'Final estimate: PPL =' ppl.${model}_${imatrix}) -- HellaSwag: Score = $(grep -Po "(?<=^${HELLASWAG_NTASK}\s).+" hellaswag.${OUTPUT_NAME})" >> "$OUTPUT_SUMMARY"

        exit

    done
done

exit 0

for prec in 08 09 10 11 12 13; do
    echo $prec
    export ZFP_PREC=$prec

    OUTPUT_NAME="from_ZFP-PREC_${ZFP_PREC}_dim_${DIM}"

    ./_build/bin/llama-perplexity ${SETTINGS} -m ${PREFIX}-F16_${OUTPUT_NAME}.gguf 2>&1 | tee ppl.${OUTPUT_NAME}
    echo "${OUTPUT_NAME} -- $(grep 'Final estimate: PPL =' ppl.${OUTPUT_NAME})" >> $OUTPUT_SUMMARY
done

for tol in 0.01 0.10 0.12 0.13 ; do
    echo $tol
    export ZFP_TOL=$tol

    OUTPUT_NAME="from_ZFP-TOL_${ZFP_TOL}_dim_${DIM}"

    ./_build/bin/llama-perplexity ${SETTINGS} -m ${PREFIX}-F16_${OUTPUT_NAME}.gguf 2>&1 | tee ppl.${OUTPUT_NAME}
    echo "${OUTPUT_NAME} -- $(grep 'Final estimate: PPL =' ppl.${OUTPUT_NAME})" >> $OUTPUT_SUMMARY
done

cd -



##### New
'

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

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
    models=( "3-8B" )
    search_command="find \${MODEL_SOURCEDIR}/weights_F16 -type f -size +60G| head -n 1"
else
    models=( "3-8B" "3-70B" "3.1-8B" "3.1-70B" )
    search_command="find \${MODEL_SOURCEDIR}/weights_F16 -type f -size +60G| head -n 1"
fi


for model in "${models[@]}"; do

    MODEL_SOURCEDIR="${SOURCE_DIR}Meta-Llama-${model}"
    MODEL_PREFIX="${MODEL_SOURCEDIR}/Meta-Llama-${model}"
    
    NGPUS=$([[ $model =~ 70B ]] && echo "2" || echo "1" )
    
    if [[ "${1:-}" == "test" ]]; then
        search_command="find ${MODEL_SOURCEDIR}/weights_F16 -type f -size +60G| head -n 1"
    else
        search_command="find ${MODEL_SOURCEDIR}/weights_F16 -type f -size +60G| head -n 1"
    fi

    for GGUF_F16_FILE in $(eval "$search_command") ; do
        OUTPUT_NAME_="$(basename -- "$GGUF_F16_FILE")"
        
        OUTPUT_NAME="${OUTPUT_NAME_%.gguf}"
        
        echo "Found Model: ${GGUF_F16_FILE}"
        
        mkdir -p ${MODEL_SOURCEDIR}/jobs_eval
        mkdir -p ${MODEL_SOURCEDIR}/logs_eval
        
        JOB_SCRIPT="${MODEL_SOURCEDIR}/jobs_eval/job_script_${model}_${OUTPUT_NAME}.sh"

        cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c ${NCPUS}
#SBATCH --mem=200G
#SBATCH -A p_darwin
#SBATCH --output=${MODEL_SOURCEDIR}/logs_eval/log.${OUTPUT_NAME}_%j.out
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:${NGPUS}

cd ${EXEC_DIR}

module purge
source ./modules.rc

srun ${EXECUTABLE_CLI} \
     ${SETTINGS} \
     -m $GGUF_F16_FILE \
     --repeat_penalty 1.0 \
     --prompt "Tell me a German joke. I've never heard one before." \
     --predict 200 \
     2>&1 | tee ${MODEL_SOURCEDIR}/logs_eval/cli.${OUTPUT_NAME}

srun ${EXECUTABLE_PPL} \
     ${SETTINGS} \
     --hellaswag \
     -f "${SOURCE_DIR}/hellaswag_val_full.txt" \
     --hellaswag-tasks ${HELLASWAG_NTASK} \
     -m $GGUF_F16_FILE \
     2>&1 | tee ${MODEL_SOURCEDIR}/logs_eval/hellaswag.${OUTPUT_NAME}

srun ${EXECUTABLE_PPL} \
     ${SETTINGS} \
     --perplexity \
     --file ${SCRIPT_DIR}/wiki.train.raw \
     -m $GGUF_F16_FILE \
     2>&1 | tee ${MODEL_SOURCEDIR}/logs_eval/ppl.${OUTPUT_NAME}


PPL_RESULT=$(grep 'Final estimate: PPL =' ppl.${OUTPUT_NAME} || true)
HSWAG_RESULT=$(grep -Po "(?<=^${HELLASWAG_NTASK}[[:space:]]).+" hellaswag.${OUTPUT_NAME} || true)

echo "${OUTPUT_NAME} -- ${PPL_RESULT:-N/A} -- HellaSwag: Score = ${HSWAG_RESULT:-N/A}" >> "$OUTPUT_SUMMARY"

EOF
        sync
        sleep 0.05
        sbatch "$JOB_SCRIPT"


    done # gguf-file
done # model 