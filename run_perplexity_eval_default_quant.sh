#!/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=70G
#SBATCH -A p_darwin
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1

SOURCE_DIR="/data/horse/ws/s0872522-llm-zfp/llama.cpp/"
PREFIX="${SOURCE_DIR}Meta-Llama-3-8B/Meta-Llama-3-8B"
EXEC_DIR="/home/s0872522/workspaces/cat/s0872522-llm-zfp/llama.cpp"

cd $EXEC_DIR

source ./modules.rc

OUTPUT_SUMMARY="${SOURCE_DIR}Meta-Llama-3-8B/metric.summary_default_quant"

echo "" > $OUTPUT_SUMMARY

DIM="4"
export NCPUS=8
HELLASWAG_NTASK=1000
OUTPUT_QUANTIZATIONS=(BF16 F16 F32 IQ1_M IQ1_S IQ2_M IQ2_S IQ2_XS IQ2_XXS IQ3_M IQ3_S IQ3_XS IQ3_XXS IQ4_NL IQ4_XS Q2_K Q2_K_S Q3_K_L Q3_K_M Q3_K_S Q4_0 Q4_0_4_4 Q4_0_4_8 Q4_0_8_8 Q4_1  Q4_K_M Q4_K_S Q5_0 Q5_1 Q5_K_M Q5_K_S Q6_K Q8_0 TQ1_0 TQ2_0)

SETTINGS="-ngl 300 -s 1 -t ${NCPUS} --ctx-size 4096 "

for imatrix in noimatrix withimatrix ; do

	for model in "${OUTPUT_QUANTIZATIONS[@]}" ; do

	    srun ./_build/bin/llama-perplexity ${SETTINGS} --perplexity --file ./Meta-Llama-3-8B/wiki.train.raw -m "${PREFIX}-F16_${model}_${imatrix}.gguf" 2>&1 | tee ppl.${model}_${imatrix}
	    srun ./_build/bin/llama-perplexity ${SETTINGS} --hellaswag -f "${SOURCE_DIR}/hellaswag_val_full.txt" --hellaswag-tasks ${HELLASWAG_NTASK} -m "${PREFIX}-F16_${model}_${imatrix}.gguf" 2>&1 | tee hellaswag.${model}_${imatrix}
	    echo "${model}_${imatrix} -- $(grep 'Final estimate: PPL =' ppl.${model}_${imatrix}) -- HellaSwag: Score = $(grep -Po "(?<=^${HELLASWAG_NTASK}\s).+" hellaswag.${model}_${imatrix})" >> "$OUTPUT_SUMMARY"

	done

done

cd -
