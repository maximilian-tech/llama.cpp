#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -A p_lv_scc25
#SBATCH --time=03:00:00
#SBATCH --hint=multithread

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

module purge

source $SCRIPT_DIR/../source_env_llvm.rc

echo "<--> Starting Compile Default"

ZFP=OFF ./run_compile.sh

echo "<--> Starting Compile Rate"
./run_compile.sh rate

echo "<--> Starting Compile Acc"
./run_compile.sh acc

echo "<--> Starting Compile Prec"
./run_compile.sh prec

echo "<--> Ending Compile"