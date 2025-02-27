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

SOURCE_DIR="/data/horse/ws/s0872522-llm-zfp/llama.cpp/"
EXEC_DIR="/home/s0872522/workspaces/cat/s0872522-llm-zfp/llama.cpp"

PREFIX="./Meta-Llama-3-8B/Meta-Llama-3-8B"
export NCPUS=1

SOURCE_TYPE="F16"
models=( "3-8B" "3-70B" "3.1-8B" "3.1-70B" )

OUTPUT_SUMMARY=log.summary
echo "" > $OUTPUT_SUMMARY

# gdb -batch -ex "run" -ex "bt" --args 

for mode in rate prec acc ; done
    for model in "${models[@]}" ; do
        for imatrix in wi_imat no_imat ; do
            for DIM in 4 3 2 1 ; do
                if [[ $mode == "rate"]] ; do
                    export PARAMETERS=( 3.50 4.00 4.50 4.65 5.00 6.00 8.00 )
                elif [[ $mode == "prec"]] ; do
                    export PARAMETERS=( 08 09 10 11 12 13 )
                elif [[ $mode == "acc"]] ; do
                    export PARAMETERS=( 0.05 0.10 0.12 0.13 0.14 )
                else
                    echo "Unknown mode '$mode'"
                done
                for PARAMETER  in "${PARAMETERS[@]}" ; do # Higher is better
                    if [[ $mode == "rate"]] ; do
                        if [[ $imatrix == "wi_imat" ]] ;do    
                            export ZFP_RATE_MIN=$(echo "$PARAMETER" | bc | awk '{printf "%.2f\n", $0}')
                            export ZFP_RATE_MAX=$(echo "$PARAMETER + 4" | awk '{printf "%.2f\n", $0}')
                        elif [[ $imatrix == "no_imat" ]] ;do    
                            export ZFP_RATE=$PARAMETER
                            export ZFP_RATE_MIN=$ZFP_RATE
                            export ZFP_RATE_MAX=$ZFP_RATE
                        else 
                            echo "Unknown imatrix value '$imatrix'"; exit 1
                        fi
                        export VALUE_MIN=$ZFP_RATE_MIN
                        export VALUE_MAX=$ZFP_RATE_MAX
                    elif [[ $mode == "prec"]] ; do
                        if [[ $imatrix == "wi_imat" ]] ;do    
                            echo "Precision: '$prec'"
                            export ZFP_PREC_MIN=$(echo "$prec - 2" | bc | awk '{printf "%02d\n", $0}')
                            export ZFP_PREC_MAX=$prec
                        elif [[ $imatrix == "no_imat" ]] ;do    
                            export ZFP_PREC=$PARAMETER
                            export ZFP_PREC_MIN=$ZFP_PREC
                            export ZFP_PREC_MAX=$ZFP_PREC
                        else 
                            echo "Unknown imatrix value '$imatrix'"; exit 1
                        fi
                        export VALUE_MIN=$ZFP_PREC_MIN
                        export VALUE_MAX=$ZFP_PREC_MAX
                    elif [[ $mode == "acc"]] ; do
                        if [[ $imatrix == "wi_imat" ]] ;do    
                            echo "Tolerance: '$tol'"
                            export ZFP_TOL_MIN=$(echo "$tol - 0.02" | bc | awk '{printf "%.2f\n", $0}')
                            export ZFP_TOL_MAX=$tol
                        elif [[ $imatrix == "no_imat" ]] ;do    
                            export ZFP_TOL=$PARAMETER
                            export ZFP_TOL_MIN=$ZFP_TOL
                            export ZFP_TOL_MAX=$ZFP_TOL
                        else 
                            echo "Unknown imatrix value '$imatrix'"; exit 1
                        fi
                        export VALUE_MIN=$ZFP_TOL_MIN
                        export VALUE_MAX=$ZFP_TOL_MAX
                    fi
                    
                    export OUTPUT_NAME="from_ZFP-${mode}_${VALUE_MIN}-${VALUE_MAX}_dim_${DIM}"
                    
                    MODEL_SOURCEDIR="${SOURCE_DIR}Meta-Llama-${model}"
                    MODEL_PREFIX="${MODEL_SOURCEDIR}/Meta-Llama-${model}"    
                    
                    srun ./build/bin/llama-quantize.${mode}.${imatrix}.dim_${DIM} \
                        --imatrix ${PREFIX}-imatrix.dat ${MODEL_PREFIX}-${SOURCE_TYPE}.gguf \
                        ${MODEL_PREFIX}-ZFP_tmp.gguf \
                        ZFP \
                        ${NCPUS} \
                        | tee log.${OUTPUT_NAME}
                    
                    srun ./build/bin/llama-quantize.${mode}.${imatrix}.dim_${DIM} \
                        --allow-requantize \
                        ${MODEL_PREFIX}-ZFP_tmp.gguf \
                        ${MODEL_PREFIX}-${SOURCE_TYPE}_${OUTPUT_NAME}.gguf \
                        ${SOURCE_TYPE} \
                        ${NCPUS}            

                    grep "^ZFP_RESULT" log.${OUTPUT_NAME} >> $OUTPUT_SUMMARY
                done # parameter
            done # dim 
        done # imat
    done # model
done # type