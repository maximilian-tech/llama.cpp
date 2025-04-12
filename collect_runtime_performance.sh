#!/bin/bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "${SCRIPT_DIR}"

SOURCE_DIR="/data/horse/ws/s0872522-llm-zfp/llama.cpp/"

models=( "3.1-8B" )



for model in "${models[@]}"; do

    MODEL_SOURCEDIR="${SOURCE_DIR}Meta-Llama-${model}"
    MODEL_PREFIX="${MODEL_SOURCEDIR}/Meta-Llama-${model}"

    OUTPUT_CSV=${MODEL_SOURCEDIR}/log.runtime_performance

    #LOG_FILE=${MODEL_SOURCEDIR}/results_runtime_performance/${OUTPUT_NAME}_i\${i}.cli

    for logfile in ${MODEL_SOURCEDIR}/logs_eval_performance/* ; do

        prompt_eval_time=$( tac "$logfile" | grep "prompt eval time"  | grep -oP '\d+,\d+(?= tokens per second)'  | tr ',' '.' |  tr -d '\n' )
        eval_time=$( tac "$logfile" | grep "  eval time" | grep -oP '\d+,\d+(?= tokens per second)' | tr ',' '.' | tr -d '\n' )
        
        prompt_eval_time_rt=$( tac "$logfile" | grep "prompt eval time"  | grep -oP '\d+,\d+(?= ms per token)'  | tr ',' '.' |  tr -d '\n' )
        eval_time_rt=$( tac "$logfile" | grep "  eval time" | grep -oP '\d+,\d+(?= ms per token)' | tr ',' '.' | tr -d '\n' )

        
                # Output to CSV (appenqd mode)

        OUTPUT_NAME=$(basename -- "$(grep -oP '(?<=^LOG_FILE).*' ${logfile} )" .cli)
        
        echo "${OUTPUT_NAME}"
        
        NODE=$(grep -oP '(?<=^Node: ).*' ${logfile} )
        echo "${NODE}"
        echo "${OUTPUT_NAME};node;${NODE};token_per_s_eval;${prompt_eval_time};token_per_s_gen;${eval_time};ms_per_token_eval;${prompt_eval_time_rt};ms_per_token_gen;${eval_time_rt}" >> "$OUTPUT_CSV"
        #echo $logfile

    done

done