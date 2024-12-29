#!/bin/env bash
ZFP_MODE="RATE"
[ "$1" = "rate" ] && ZFP_MODE="RATE"
[ "$1" = "acc" ] && ZFP_MODE="ACCURACY"
[ "$1" = "prec" ] && ZFP_MODE="PRECISION"

set -euo pipefail

#SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--instrument-filter="
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--thread=pthread --instrument-filter=$PWD/initial_scorep_llvm.filter"
for dim in 1 2 3 4 ; do
	SCOREP_WRAPPER=OFF cmake \
		-G Ninja \
		-B build \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=False \
		-DGGML_LTO=True \
		-DGGML_NATIVE=True \
		-DGGML_AVX512=ON \
		-DGGML_AVX512_VBMI=True \
		-DGGML_AVX512_VNNI=ON \
		-DGGML_AVX512_BF16=ON \
		-DGGML_ZFP_ENABLE=ON \
		-DBUILD_UTILITIES=OFF \
		-DZFP_WITH_OPENMP=OFF \
		-DCMAKE_C_FLAGS_RELEASE="  -O3 -DNDEBUG -march=native -flto=full -mprefer-vector-width=512 -g -gdwarf-4 -fno-omit-frame-pointer" \
		-DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -flto=full -mprefer-vector-width=512 -g -gdwarf-4 -fno-omit-frame-pointer" \
		-DZFP_ENABLE_PIC=OFF \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DCMAKE_LINKER_TYPE=LLD \
		-DGGML_ZFP_MODE=${ZFP_MODE} \
		-DGGML_ZFP_DIMENSION=${dim} \
		--fresh #\
	#	-DCMAKE_C_COMPILER=scorep-clang \
	#	-DCMAKE_CXX_COMPILER=scorep-clang++ 

	cmake --build build
	#pushd build
done
#	make -j 8 -B
#popd


