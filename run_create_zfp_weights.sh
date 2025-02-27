#!/bin/env bash

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

SOURCE_TYPE="F16"

if [[ "$1" == "test" ]]; then
    models=( "3-8B" )
    imatrizes=( wi_imat no_imat )
    dims=( 3 )
    modes=( rate )
    
    rate_parameters=( 4.00  )
    prec_parameters=( 08 09 10 11 12 13 )
    acc_parameters=( 0.05 0.10 0.12 0.13 0.14 )
else
    # default
    models=( "3-8B" "3-70B" "3.1-8B" "3.1-70B" )
    imatrizes=( wi_imat no_imat )
    dims=( 4 3 2 1 )
    modes=( rate ) #( rate prec acc )
    rate_parameters=( 3.00 3.50 4.00 4.50 5.00 6.00 8.00 )
    prec_parameters=( 08 09 10 11 12 13 )
    acc_parameters=( 0.05 0.10 0.12 0.13 0.14 )
fi




OUTPUT_SUMMARY="log.summary"
echo "" > "$OUTPUT_SUMMARY"

for mode in "${modes[@]}"; do
    for model in "${models[@]}"; do
        for imatrix in "${imatrizes[@]}"; do
            for DIM in "${dims[@]}"; do
                if [[ $mode == "rate" ]]; then
                    PARAMETERS=$rate_parameters
                elif [[ $mode == "prec" ]]; then
                    PARAMETERS=$prec_parameters
                elif [[ $mode == "acc" ]]; then
                    PARAMETERS=$acc_parameters
                else
                    echo "Unknown mode '$mode'"
                    exit 1
                fi
                
                for PARAMETER in "${PARAMETERS[@]}"; do
                    if [[ $mode == "rate" ]]; then
                        if [[ $imatrix == "wi_imat" ]]; then    
                            export ZFP_RATE_MIN=$(echo "$PARAMETER" | bc | awk '{printf "%.2f\n", $0}')
                            export ZFP_RATE_MAX=$(echo "12" | bc | awk '{printf "%.2f\n", $0}')
                        elif [[ $imatrix == "no_imat" ]]; then    
                            export ZFP_RATE=$PARAMETER
                            export ZFP_RATE_MIN=$ZFP_RATE
                            export ZFP_RATE_MAX=$ZFP_RATE
                        else 
                            echo "Unknown imatrix value '$imatrix'"; exit 1
                        fi
                        VALUE_MIN=$ZFP_RATE_MIN
                        VALUE_MAX=$ZFP_RATE_MAX
                    elif [[ $mode == "prec" ]]; then
                        if [[ $imatrix == "wi_imat" ]]; then    
                            echo "Precision: '$PARAMETER'"
                            export ZFP_PREC_MIN=$(echo "$PARAMETER - 2" | bc | awk '{printf "%02d\n", $0}')
                            export ZFP_PREC_MAX=$PARAMETER
                        elif [[ $imatrix == "no_imat" ]]; then    
                            export ZFP_PREC=$PARAMETER
                            export ZFP_PREC_MIN=$ZFP_PREC
                            export ZFP_PREC_MAX=$ZFP_PREC
                        else 
                            echo "Unknown imatrix value '$imatrix'"; exit 1
                        fi
                        VALUE_MIN=$ZFP_PREC_MIN
                        VALUE_MAX=$ZFP_PREC_MAX
                    elif [[ $mode == "acc" ]]; then
                        if [[ $imatrix == "wi_imat" ]]; then    
                            echo "Tolerance: '$PARAMETER'"
                            export ZFP_TOL_MIN=$(echo "$PARAMETER - 0.02" | bc | awk '{printf "%.2f\n", $0}')
                            export ZFP_TOL_MAX=$PARAMETER
                        elif [[ $imatrix == "no_imat" ]]; then    
                            export ZFP_TOL=$PARAMETER
                            export ZFP_TOL_MIN=$ZFP_TOL
                            export ZFP_TOL_MAX=$ZFP_TOL
                        else 
                            echo "Unknown imatrix value '$imatrix'"; exit 1
                        fi
                        VALUE_MIN=$ZFP_TOL_MIN
                        VALUE_MAX=$ZFP_TOL_MAX
                    fi
                    
                    OUTPUT_NAME="from_ZFP-${mode}_${VALUE_MIN}-${VALUE_MAX}_${imatrix}_dim_${DIM}"
                    
                    MODEL_SOURCEDIR="${SCRIPT_DIR}/Meta-Llama-${model}"
                    MODEL_PREFIX="Meta-Llama-${model}"
                    
                    EXECUTABLE="${SCRIPT_DIR}/build/bin/llama-quantize.${mode}.${imatrix}.dim_${DIM}"
                    if [[ ! -x "$EXECUTABLE" ]]; then
                        echo "Error: Executable not found: $EXECUTABLE"
                        exit 1
                    fi

                    if [[ "$imatrix" == "wi_imat" ]]; then
                        IMATRIX_OPTION="--imatrix ${MODEL_SOURCEDIR}/imatrix.dat"
                    else
                        IMATRIX_OPTION=""
                    fi


                    mkdir -p ${MODEL_SOURCEDIR}/jobs
                    mkdir -p ${MODEL_SOURCEDIR}/logs
                    mkdir -p ${MODEL_SOURCEDIR}/zfp_tmp
                    mkdir -p ${MODEL_SOURCEDIR}/zfp
                    
                    JOB_SCRIPT="${MODEL_SOURCEDIR}/jobs/job_script_${model}_${OUTPUT_NAME}.sh"

                    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --output=${MODEL_SOURCEDIR}/logs/log.${OUTPUT_NAME}_%j.out
#SBATCH --mem=10G
#SBATCH -A p_lv_scc25
#SBATCH --time=03:00:00
#SBATCH --hint=multithread

cat $0

env | grep ZFP

module purge
module load $SCRIPT_DIR/../source_env_llvm.rc

set -euo pipefail

time srun "$EXECUTABLE" \
    ${IMATRIX_OPTION} \
    "${MODEL_SOURCEDIR}/${MODEL_PREFIX}-${SOURCE_TYPE}.gguf" \
    "${MODEL_SOURCEDIR}/zfp_tmp/${MODEL_PREFIX}-ZFP_${OUTPUT_NAME}.gguf" \
    ZFP \
    \${SLURM_CPUS_PER_TASK} \
    | tee >( grep "^ZFP_RESULT" > "${MODEL_SOURCEDIR}/log.${OUTPUT_NAME}")

time srun "$EXECUTABLE" \
    --allow-requantize \
    "${MODEL_SOURCEDIR}/zfp_tmp/${MODEL_PREFIX}-ZFP_${OUTPUT_NAME}.gguf" \
    "${MODEL_SOURCEDIR}/zfp/${MODEL_PREFIX}-${SOURCE_TYPE}_${OUTPUT_NAME}.gguf" \
    "${SOURCE_TYPE}" \
    \${SLURM_CPUS_PER_TASK}            

grep "^ZFP_RESULT" "${MODEL_SOURCEDIR}/log.${OUTPUT_NAME}" >> "$OUTPUT_SUMMARY"
rm "${MODEL_SOURCEDIR}/log.${OUTPUT_NAME}"


EOF

                    sbatch "$JOB_SCRIPT"
                done # parameter
            done # dim 
        done # imat
    done # model
done # mode
