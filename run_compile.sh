#!/bin/env bash

set -euo pipefail

#SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--instrument-filter="
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--thread=pthread --instrument-filter=$PWD/initial_scorep_llvm.filter"

SCOREP_WRAPPER=OFF cmake \
	-G Ninja \
	-B build \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DBUILD_SHARED_LIBS=False \
	-DGGML_LTO=True \
	-DGGML_NATIVE=True \
	-DGGML_AVX512=ON \
	-DGGML_AVX512_VBMI=True \
	-DGGML_AVX512_VNNI=ON \
	-DGGML_AVX512_BF16=ON \
	-DGGML_ZFP=ON \
	-DBUILD_UTILITIES=OFF \
	-DZFP_WITH_OPENMP=OFF \
	-DCMAKE_C_FLAGS_RELEASE="  -O3 -DNDEBUG -march=native -flto=full -mprefer-vector-width=512 -g -gdwarf-4 -fno-omit-frame-pointer" \
	-DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -flto=full -mprefer-vector-width=512 -g -gdwarf-4 -fno-omit-frame-pointer" \
	-DZFP_ENABLE_PIC=OFF \
	-DCMAKE_C_COMPILER=clang \
	-DCMAKE_CXX_COMPILER=clang++ \
	-DCMAKE_LINKER_TYPE=LLD \
	--fresh #\
#	-DCMAKE_C_COMPILER=scorep-clang \
#	-DCMAKE_CXX_COMPILER=scorep-clang++ 

cmake --build build 
#pushd build

#	make -j 8 -B
#popd


