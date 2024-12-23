#!/bin/env bash



OUTPUT_SUMMARY=ppl.summary

echo "" > $OUTPUT_SUMMARY

PREFIX="/data/horse/ws/s0872522-llm-zfp/llama.cpp/Meta-Llama-3-8B/Meta-Llama-3-8B"
#PREFIX="./Meta-Llama-3-8B/Meta-Llama-3-8B"
DIM="4"
export NCPUS=8

SETTINGS="-ngl 300 --ctx-size 4096 -s 1 -t ${NCPUS} --perplexity --file ./Meta-Llama-3-8B/wiki.train.raw"

for model in F16 Q8_K Q4_K ; do

    ./_build/bin/llama-perplexity ${SETTINGS} -m ${PREFIX}-${model}.gguf  2>&1 | tee ppl.${model}
    echo "${model} -- $(grep 'Final estimate: PPL =' ppl.${model})" >> "$OUTPUT_SUMMARY"

done

for rate in 3.5 4.0 4.5 4.65 5.0 6.0 8.0 ; do
    echo $rate
    export ZFP_RATE=$rate
    
    OUTPUT_NAME="from_ZFPRATE_${ZFP_RATE}_dim_${DIM}"
    
    ./_build/bin/llama-perplexity ${SETTINGS} -m ${PREFIX}-F16_${OUTPUT_NAME}.gguf 2>&1 | tee ppl.${OUTPUT_NAME}
    echo "${OUTPUT_NAME} -- $(grep 'Final estimate: PPL =' ppl.${OUTPUT_NAME})" >> $OUTPUT_SUMMARY
done


for tol in 0.01 0.1 0.12 0.13 ; do
    echo $tol
    export ZFP_TOL=$tol
    OUTPUT_NAME="from_ZFPTOL_${ZFP_TOL}_dim_${DIM}"

    ./_build/bin/llama-perplexity ${SETTINGS} -m ${PREFIX}-F16_${OUTPUT_NAME}.gguf 2>&1 | tee ppl.${OUTPUT_NAME}
    echo "${OUTPUT_NAME} -- $(grep 'Final estimate: PPL =' ppl.${OUTPUT_NAME})" >> $OUTPUT_SUMMARY
done
