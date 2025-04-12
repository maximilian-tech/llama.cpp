#!/bin/env bash

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

SOURCE_DIR="/data/horse/ws/s0872522-llm-zfp/llama.cpp/"

#CLI_PROMPT="How much wood would a woodchuck chuck if a woodchuck could chuck wood?"

# Approx. 154 Tokens
CLI_PROMPT="How much wood would a woodchuck chuck if a woodchuck could chuck wood? This age-old tongue twister has puzzled many, but let’s explore it from multiple angles. Scientifically, a woodchuck (or groundhog) doesn’t actually chuck wood, but if it could, we might estimate its capabilities based on its burrowing behavior. \
            According to a study, a woodchuck moves roughly 700 pounds of dirt when digging a burrow. If we equate this to wood, we might assume a woodchuck could chuck a similar amount. However, the physics of woodchucking would depend on its bite force, jaw strength, and endurance. Could it sustain wood-chucking for long durations, or would it tire quickly?"


if [[ "${1:-}" == "test" ]]; then
    models=( "3.1-8B" )
    modes=(
        #ZFP_from_ZFP-rate_4.00-4.00_no_imat_dim_3
        Q8_0_no_imat
    )
    cores=( 96 )
else
    #models=( "3-8B" "3-70B" "3.1-8B" "3.1-70B" )
    models=( "3.1-8B" )
    cores=( 24 48 96 )
    modes=(
        ZFP_from_ZFP-rate_4.00-4.00_no_imat_dim_3
        ZFP_from_ZFP-rate_4.00-4.00_no_imat_dim_4
        ZFP_from_ZFP-rate_6.00-6.00_no_imat_dim_3
        ZFP_from_ZFP-rate_6.00-6.00_no_imat_dim_4
        ZFP_from_ZFP-rate_8.00-8.00_no_imat_dim_3
        ZFP_from_ZFP-rate_8.00-8.00_no_imat_dim_4
        Q4_0_no_imat
        Q4_1_no_imat
        Q4_K_M_no_imat
        Q4_K_S_no_imat
        Q5_1_no_imat
        Q8_0_no_imat
    )

fi



for model in "${models[@]}"; do

    MODEL_SOURCEDIR="${SOURCE_DIR}Meta-Llama-${model}"
    MODEL_PREFIX="${MODEL_SOURCEDIR}/Meta-Llama-${model}"

    OUTPUT_CSV=${MODEL_SOURCEDIR}/log.runtime_performance

    for GGUF_F16_FILE_ABBR in "${modes[@]}" ; do
        
        OUTPUT_NAME="${model}_${GGUF_F16_FILE_ABBR}"
        
        echo "Found Model: ${GGUF_F16_FILE_ABBR}"
        
        GGUF_F16_FILE=${MODEL_SOURCEDIR}/weights/Meta-Llama-${model}-${GGUF_F16_FILE_ABBR}.gguf
        if [ ! -f ${GGUF_F16_FILE} ]; then
            echo "File ${GGUF_F16_FILE} not found!"
            exit 2
        fi

        mkdir -p "${MODEL_SOURCEDIR}/jobs_eval_performance"
        mkdir -p "${MODEL_SOURCEDIR}/logs_eval_performance"
        mkdir -p "${MODEL_SOURCEDIR}/results_runtime_performance"
        
        echo "GGUF_F16_FILE_ABBR=${GGUF_F16_FILE_ABBR}"
                
        if [[ "${GGUF_F16_FILE_ABBR}" =~ .*dim_3$ ]]; then
            EXECUTABLE_CLI="${SOURCE_DIR}/build/bin/llama-cli.rate.no_imat.dim_3"
        elif [[ "${GGUF_F16_FILE_ABBR}" =~ .*dim_4$ ]]; then
            EXECUTABLE_CLI="${SOURCE_DIR}/build/bin/llama-cli.rate.no_imat.dim_4"
        elif [[ "${GGUF_F16_FILE_ABBR}" =~ .*no_imat$ ]]; then
            EXECUTABLE_CLI="${SOURCE_DIR}/build/bin/llama-cli"
        fi
    
        for NCPUS in "${cores[@]}" ; do
            for i in {1..7}; do
                JOB_SCRIPT="${MODEL_SOURCEDIR}/jobs_eval_performance/job_script_${OUTPUT_NAME}_n${NCPUS}_i${i}.sh"
                
                cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 104
#SBATCH --mem=200G
#SBATCH -A zihforschung
#SBATCH --output="${MODEL_SOURCEDIR}/logs_eval_performance/log.${OUTPUT_NAME}_n${NCPUS}_i${i}_%j.out"
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --constraint=no_monitoring
##SBATCH --reservation=p_lv_scc25_432
#SBATCH --cpu-freq=2000000

cat \$0

cd "${SCRIPT_DIR}"

module purge
source ${SCRIPT_DIR}/../source_env_llvm.rc

export OMP_NUM_THREADS=${NCPUS}
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread



LOG_FILE=${MODEL_SOURCEDIR}/results_runtime_performance/${OUTPUT_NAME}_n${NCPUS}_i${i}.cli

NODE_NAME=\$(srun hostname)
echo "Node: \${NODE_NAME}"

time srun --cpu-bind=cores -c 104 -- \
    "${EXECUTABLE_CLI}" \
    -s 1 \
    -t ${NCPUS} \
    --ctx-size 4096 \
    -m "${GGUF_F16_FILE}" \
    --repeat_penalty 1.0 \
    --prompt "${CLI_PROMPT}" \
    --predict 200 \
    --ignore-eos \
    --no-mmap \
    2>&1 | tee \${LOG_FILE}
    
sync \${LOG_FILE}

# Extract values using grep and awk
prompt_eval_time=\$(grep "prompt eval time" "\$LOG_FILE" | awk '{print \$16}' | tr -d '\n' )
eval_time=\$(grep "eval time" "\$LOG_FILE" | awk '{print \$15}' | tr -d '\n' )

# Output to CSV (append mode)
echo "${OUTPUT_NAME},ncores,${NCPUS},iteration,${i},node,\${NODE_NAME},token_per_s_eval,\${prompt_eval_time},token_per_s_gen,\${eval_time}" >> "$OUTPUT_CSV"


EOF
            sync
            #sleep 0.05
            #sbatch "$JOB_SCRIPT"
            done #iteration
        done # cores
    done # gguf-file
done # model 