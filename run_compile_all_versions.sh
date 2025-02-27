#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

set -euo pipefail

echo "<--> Starting Compile Default"

ZFP=OFF ./run_compile.sh

echo "<--> Starting Compile Rate"
./run_compile.sh rate

echo "<--> Starting Compile Acc"
./run_compile.sh acc

echo "<--> Starting Compile Prec"
./run_compile.sh prec

echo "<--> Ending Compile"