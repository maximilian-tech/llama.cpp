#!/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=50G
#SBATCH -A p_darwin
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread

source ../source_env_llvm.rc

set -euxo pipefail

GG="Meta-Llama-3-8B/Meta-Llama-3-8B"
I="F32"
for O in Q4_0 Q4_1 Q5_0 Q5_1 IQ2_M TQ1_0 TQ2_0 Q2_K IQ3_XXS IQ3_S IQ3_M Q3_K IQ3_XS Q3_K_S Q3_K_M Q3_K_L IQ4_NL IQ4_XS Q4_K Q4_K_S Q4_K_M Q5_K Q5_K_S Q5_K_M Q6_K Q8_0 Q4_0_4_4 Q4_0_4_8 Q4_0_8_8 F16 BF16 IQ1_S IQ1_M IQ2_S IQ2_XXS IQ2_XS Q2_K_S ZFP; do
  rm -f "${GG}-${O}-noimatrix.gguf"
  ./build/bin/llama-quantize "${GG}-${I}.gguf" "${GG}-${O}-noimatrix.gguf" ${O} $(nproc) \
    2>&1 | tee -a "${GG}.log"
done

exit 4

for O in Q4_0 Q4_1 Q5_0 Q5_1 IQ2_M TQ1_0 TQ2_0 Q2_K IQ3_XXS IQ3_S IQ3_M Q3_K IQ3_XS Q3_K_S Q3_K_M Q3_K_L IQ4_NL IQ4_XS Q4_K Q4_K_S Q4_K_M Q5_K Q5_K_S Q5_K_M Q6_K Q8_0 Q4_0_4_4 Q4_0_4_8 Q4_0_8_8 F16 BF16 IQ1_S IQ1_M IQ2_S IQ2_XXS IQ2_XS Q2_K_S ZFP; do
  rm -f "${GG}-${O}-withimatrix.gguf"
  ./build/bin/llama-quantize --imatrix "${GG}-imatrix.dat" "${GG}-${I}.gguf" "${GG}-${O}-withimatrix.gguf" ${O} $(nproc) \
    2>&1 | tee -a "${GG}.log"
done