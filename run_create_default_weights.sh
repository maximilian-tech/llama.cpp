#!/bin/env bash
# run_create_default_weights.sh

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

SOURCE_TYPE="F16"

if [[ "${1:-}" == "test" ]]; then
    models=( "3-8B" )
    imatrizes=( wi_imat no_imat )
    modes=( Q4_0 )

else
    # default
    #models=( "3-8B" "3-70B" "3.1-8B" "3.1-70B" )
    models=( "3.1-8B" "3.1-70B" )
    imatrizes=( wi_imat no_imat )
    
    modes=( 
            Q4_0 
            Q4_1 
            Q5_0 
            Q5_1 
            IQ2_M 
            TQ1_0
            TQ2_0
            Q2_K
            Q2_K_S
            IQ3_XXS
            IQ3_S
            IQ3_M
            IQ3_XS
            Q3_K_S
            Q3_K_M
            Q3_K_L
            IQ4_NL
            IQ4_XS
            Q4_K_S
            Q4_K_M
            Q5_K_S
            Q5_K_M
            Q6_K
            Q8_0
            F16
            BF16
            IQ1_S
            IQ1_M
            IQ2_S
            IQ2_XXS
            IQ2_XS
          )
fi

            # Q4_0_4_4
            # Q4_0_4_8
            # Q4_0_8_8


for mode in "${modes[@]}"; do
    for model in "${models[@]}"; do
        for imatrix in "${imatrizes[@]}"; do

                    OUTPUT_NAME="${mode}_${imatrix}"
                    
                    MODEL_SOURCEDIR="${SCRIPT_DIR}/Meta-Llama-${model}"
                    MODEL_PREFIX="Meta-Llama-${model}"
                    
                    EXECUTABLE="${SCRIPT_DIR}/build/bin/llama-quantize"
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
                    mkdir -p ${MODEL_SOURCEDIR}/result_quant
                    mkdir -p ${MODEL_SOURCEDIR}/weights
                    mkdir -p ${MODEL_SOURCEDIR}/weights_F16
                    
                    OUTPUT_SUMMARY="${MODEL_SOURCEDIR}/log.quant"

                    JOB_SCRIPT="${MODEL_SOURCEDIR}/jobs/job_script_${model}_${OUTPUT_NAME}.sh"

                    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --output=${MODEL_SOURCEDIR}/logs/log.${OUTPUT_NAME}_%j.out
#SBATCH --mem=40G
#SBATCH -A p_lv_scc25
#SBATCH --time=05:00:00
#SBATCH --hint=multithread

cat $JOB_SCRIPT

module purge
source $SCRIPT_DIR/../source_env_llvm.rc

set -euo pipefail

time srun "$EXECUTABLE" \
    ${IMATRIX_OPTION} \
    "${MODEL_SOURCEDIR}/${MODEL_PREFIX}-${SOURCE_TYPE}.gguf" \
    "${MODEL_SOURCEDIR}/weights/${MODEL_PREFIX}-${OUTPUT_NAME}.gguf" \
    ${mode} \
    \${SLURM_CPUS_PER_TASK} \
    | tee >( grep "^QUANT_RESULT" > "${MODEL_SOURCEDIR}/result_quant/log.${OUTPUT_NAME}")

time srun "$EXECUTABLE" \
    --allow-requantize \
    "${MODEL_SOURCEDIR}/weights/${MODEL_PREFIX}-${OUTPUT_NAME}.gguf" \
    "${MODEL_SOURCEDIR}/weights_F16/${MODEL_PREFIX}-${SOURCE_TYPE}_${OUTPUT_NAME}.gguf" \
    "${SOURCE_TYPE}" \
    \${SLURM_CPUS_PER_TASK}            

grep "^QUANT_RESULT" "${MODEL_SOURCEDIR}/result_quant/log.${OUTPUT_NAME}" >> "$OUTPUT_SUMMARY"


EOF
                    #sleep 0.05
                    sync "$JOB_SCRIPT"
                    #sbatch "$JOB_SCRIPT"
        done # imat
    done # model
done # mode
