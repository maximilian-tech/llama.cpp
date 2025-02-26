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
export NCPUS=1

OUTPUT_SUMMARY=log.summary
echo "" > $OUTPUT_SUMMARY

# gdb -batch -ex "run" -ex "bt" --args 

#for DIM in 4 3 2 1 ; do
for DIM in 3 ; do
    #for rate in 3.50 4.00 4.50 4.65 5.00 6.00 8.00 ; do # Higher is better
    for rate in 6.0 ; do

        echo "'Rate: '$rate'"
: '        
        export ZFP_RATE_MIN=$(echo "$rate - 2" | bc | awk '{printf "%.2f\n", $0}')
        export ZFP_RATE_MAX=$(echo "$rate" | awk '{printf "%.2f\n", $0}')
        
        OUTPUT_NAME="from_ZFP-RATE_${ZFP_RATE_MIN}-${ZFP_RATE_MAX}_dim_${DIM}"

        ./build/bin/llama-quantize.rate.wi_imat.dim_${DIM} --imatrix ${PREFIX}-imatrix.dat ${PREFIX}-F16.gguf  ${PREFIX}-ZFP_tmp.gguf ZFP  ${NCPUS} | tee log.${OUTPUT_NAME}

        #./build/bin/llama-quantize.rate.dim_${DIM} ${PREFIX}-F16.gguf  ${PREFIX}-ZFP_tmp.gguf ZFP  ${NCPUS} | tee log.${OUTPUT_NAME}
        ./build/bin/llama-quantize.rate.wi_imat.dim_${DIM} --allow-requantize  ${PREFIX}-ZFP_tmp.gguf ${PREFIX}-F16_${OUTPUT_NAME}_new.gguf F16 ${NCPUS}
        grep "^ZFP_RESULT" log.${OUTPUT_NAME} >> $OUTPUT_SUMMARY
'

        export ZFP_RATE=4.10
        export ZFP_RATE_MIN=$ZFP_RATE
        export ZFP_RATE_MAX=$ZFP_RATE

        OUTPUT_NAME="from_ZFP-RATE_${ZFP_RATE_MIN}-${ZFP_RATE_MAX}_dim_${DIM}"

        ./build/bin/llama-quantize.rate.wi_imat.dim_${DIM} --imatrix ${PREFIX}-imatrix.dat ${PREFIX}-F16.gguf  ${PREFIX}-ZFP_tmp.gguf ZFP  ${NCPUS} | tee log.${OUTPUT_NAME}

        #./build/bin/llama-quantize.rate.dim_${DIM} ${PREFIX}-F16.gguf  ${PREFIX}-ZFP_tmp.gguf ZFP  ${NCPUS} | tee log.${OUTPUT_NAME}
        ./build/bin/llama-quantize.rate.wi_imat.dim_${DIM} --allow-requantize  ${PREFIX}-ZFP_tmp.gguf ${PREFIX}-F16_${OUTPUT_NAME}_new.gguf F16 ${NCPUS}
        grep "^ZFP_RESULT" log.${OUTPUT_NAME} >> $OUTPUT_SUMMARY


        exit 0

    done
#done

    for prec in 08 09 10 11 12 13; do # higher is better
        ##break
        echo "Precision: '$prec'"
        export ZFP_PREC_MIN=$(echo "$prec - 2" | bc | awk '{printf "%02d\n", $0}') # allow for better precision
        export ZFP_PREC_MAX=$prec
        OUTPUT_NAME="from_ZFP-PREC_${ZFP_PREC_MIN}-${ZFP_PREC_MAX}_dim_${DIM}"
        ./build/bin/llama-quantize.prec.dim_${DIM} ${PREFIX}-F16.gguf  ${PREFIX}-ZFP_tmp.gguf ZFP  ${NCPUS} | tee log.${OUTPUT_NAME}
        ./build/bin/llama-quantize.prec.dim_${DIM} --allow-requantize  ${PREFIX}-ZFP_tmp.gguf ${PREFIX}-F16_${OUTPUT_NAME}.gguf F16 ${NCPUS}
        grep "^ZFP_RESULT" log.${OUTPUT_NAME} >> $OUTPUT_SUMMARY
        
        break
    done

    for tol in 0.05 0.10 0.12 0.13 0.14 ; do # Higher is worse
        echo "Tolerance: '$tol'"
        export ZFP_TOL_MIN=$(echo "$tol - 0.02" | bc | awk '{printf "%.2f\n", $0}')
        export ZFP_TOL_MAX=$tol
        OUTPUT_NAME="from_ZFP-TOL_${ZFP_TOL_MIN}-${ZFP_TOL_MAX}_dim_${DIM}"
        ./build/bin/llama-quantize.acc.dim_${DIM} ${PREFIX}-F16.gguf  ${PREFIX}-ZFP_tmp.gguf ZFP  ${NCPUS} | tee log.${OUTPUT_NAME}
        ./build/bin/llama-quantize.acc.dim_${DIM} --allow-requantize  ${PREFIX}-ZFP_tmp.gguf ${PREFIX}-F16_${OUTPUT_NAME}.gguf F16 ${NCPUS}
        grep "^ZFP_RESULT" log.${OUTPUT_NAME} >> $OUTPUT_SUMMARY
        break
    done

done