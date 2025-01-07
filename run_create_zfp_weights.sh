#!/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=50G
#SBATCH -A p_darwin
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread

source ../source_env_llvm.rc

set euxo -pipefail

PREFIX="./Meta-Llama-3-8B/Meta-Llama-3-8B"
DIM="4"
export NCPUS=4

OUTPUT_SUMMARY=log.summary
echo "" > $OUTPUT_SUMMARY

for prec in 08 09 10 11 12 13; do
    echo $prec
    export ZFP_PREC=$prec
    OUTPUT_NAME="from_ZFP-PREC_${ZFP_PREC}_dim_${DIM}"
    ./build/bin/llama-quantize.prec.dim_${DIM} ${PREFIX}-F16.gguf  ${PREFIX}-ZFP_tmp.gguf ZFP  ${NCPUS} | tee log.${OUTPUT_NAME}
    ./build/bin/llama-quantize.prec.dim_${DIM} --allow-requantize  ${PREFIX}-ZFP_tmp.gguf ${PREFIX}-F16_${OUTPUT_NAME}.gguf F16 ${NCPUS}
    grep "^ZFP_RESULT" log.${OUTPUT_NAME} >> $OUTPUT_SUMMARY
done

for rate in 3.50 4.00 4.50 4.65 5.00 6.00 8.00 ; do
    echo $rate
    export ZFP_RATE=$rate
    OUTPUT_NAME="from_ZFP-RATE_${ZFP_RATE}_dim_${DIM}"
    ./build/bin/llama-quantize.rate.dim_${DIM} ${PREFIX}-F16.gguf  ${PREFIX}-ZFP_tmp.gguf ZFP  ${NCPUS} | tee log.${OUTPUT_NAME}
    ./build/bin/llama-quantize.rate.dim_${DIM} --allow-requantize  ${PREFIX}-ZFP_tmp.gguf ${PREFIX}-F16_${OUTPUT_NAME}.gguf F16 ${NCPUS}
    grep "^ZFP_RESULT" log.${OUTPUT_NAME} >> $OUTPUT_SUMMARY
done

for tol in 0.01 0.10 0.12 0.13 ; do
    echo $tol
    export ZFP_TOL=$tol
    OUTPUT_NAME="from_ZFP-TOL_${ZFP_TOL}_dim_${DIM}"
    ./build/bin/llama-quantize.acc.dim_${DIM} ${PREFIX}-F16.gguf  ${PREFIX}-ZFP_tmp.gguf ZFP  ${NCPUS} | tee log.${OUTPUT_NAME}
    ./build/bin/llama-quantize.acc.dim_${DIM} --allow-requantize  ${PREFIX}-ZFP_tmp.gguf ${PREFIX}-F16_${OUTPUT_NAME}.gguf F16 ${NCPUS}
    grep "^ZFP_RESULT" log.${OUTPUT_NAME} >> $OUTPUT_SUMMARY
done

