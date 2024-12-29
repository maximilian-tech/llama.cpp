#!/bin/env bash

set euxo -pipefail

PREFIX="./Meta-Llama-3-8B/Meta-Llama-3-8B"
DIM="4"
export NCPUS=4

OUTPUT_SUMMARY=log.summary
echo "" > $OUTPUT_SUMMARY

for prec in 8 16 32 ; do
    echo $prec
    export ZFP_PREC=$prec
    OUTPUT_NAME="from_ZFP-PREC_${ZFP_PREC}_dim_${DIM}"
    ./build/bin/llama-quantize.prec.dim_${DIM} ${PREFIX}-F16.gguf  ${PREFIX}-ZFP_tmp.gguf ZFP  ${NCPUS} | tee log.${OUTPUT_NAME}
    ./build/bin/llama-quantize.prec.dim_${DIM} --allow-requantize  ${PREFIX}-ZFP_tmp.gguf ${PREFIX}-F16_${OUTPUT_NAME}.gguf F16 ${NCPUS}
    grep "^ZFP_RESULT" log.${OUTPUT_NAME} >> $OUTPUT_SUMMARY
done

exit 1

for rate in 3.5 4.0 4.5 4.65 5.0 6.0 8.0 ; do
    echo $rate
    export ZFP_RATE=$rate
    OUTPUT_NAME="from_ZFP-RATE_${ZFP_RATE}_dim_${DIM}"
    ./build/bin/llama-quantize.rate.dim_${DIM} ${PREFIX}-F16.gguf  ${PREFIX}-ZFP_tmp.gguf ZFP  ${NCPUS} | tee log.${OUTPUT_NAME}
    ./build/bin/llama-quantize.rate.dim_${DIM} --allow-requantize  ${PREFIX}-ZFP_tmp.gguf ${PREFIX}-F16_${OUTPUT_NAME}.gguf F16 ${NCPUS}
    grep "^ZFP_RESULT" log.${OUTPUT_NAME} >> $OUTPUT_SUMMARY
done

for tol in 0.01 0.1 0.12 0.13 ; do
    echo $tol
    export ZFP_TOL=$tol
    OUTPUT_NAME="from_ZFP-TOL_${ZFP_TOL}_dim_${DIM}"
    ./build/bin/llama-quantize.acc.dim_${DIM} ${PREFIX}-F16.gguf  ${PREFIX}-ZFP_tmp.gguf ZFP  ${NCPUS} | tee log.${OUTPUT_NAME}
    ./build/bin/llama-quantize.acc.dim_${DIM} --allow-requantize  ${PREFIX}-ZFP_tmp.gguf ${PREFIX}-F16_${OUTPUT_NAME}.gguf F16 ${NCPUS}
    grep "^ZFP_RESULT" log.${OUTPUT_NAME} >> $OUTPUT_SUMMARY
done