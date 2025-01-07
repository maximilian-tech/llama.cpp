#!/bin/env bash

set euxo -pipefail
OUTPUT_SUMMARY=log.summary

for model in Q8_0 Q4_0;do
        echo $model
        SOURCE_FILE="ppl.$model"
        BPW_VALUE=`grep 'llm_load_print_meta: model size' $SOURCE_FILE | grep -oP '\(\K[^\)]*(?=\sBPW)'`
        SIZE_VALUE=$(grep 'llm_load_print_meta: model size' "$SOURCE_FILE" | grep -oP '\d+\.\d+(?=\sGiB)' | awk '{print $1 * 1024}')
        echo "NATIVE,type,native,value,$model,original_size(MiB),15317.02,compressed_size(MiB),$SIZE_VALUE,compression_ratio,XXX,bits_per_weight,$BPW_VALUE" >> $OUTPUT_SUMMARY
done

