#!/bin/env bash

PROMPT="1"
#PROMPT="Tell me a German joke. I've never heard one before."
NCORES=100
MODEL="ZFP"
#MODEL="Q8_0"
#MODEL="Q4_0"
export SCOREP_ENABLE_TRACING=True
export SCOREP_ENABLE_PROFILING=False
export SCOREP_TOTAL_MEMORY=4G

export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_NUM_THREADS=$NCORES
export OMP_MAX_ACTIVE_LEVELS=1

export SCOREP_METRIC_PLUGINS=topdown_plugin
export SCOREP_METRIC_TOPDOWN_PLUGIN='*'

perf record -g -e 'cycles,L1-dcache-load-misses' \
./build/bin/llama-cli \
	--model ./Meta-Llama-3-8B/Meta-Llama-3-8B-${MODEL}.gguf \
	--threads $NCORES \
	--repeat_penalty 1.0 \
	--prompt "$PROMPT" \
	--predict 25 \
	--seed 1
