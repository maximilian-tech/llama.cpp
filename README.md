## Goal
Replace Quantization with Floating Point Compression

## Current Problem
Very, very slow (currently).
Therefore work on performance

## ToDo
- [ ] Have working prototype according to Jens
- [ ] Integrate HDF5 + OpenBLAS in Buildsystem
- [ ] Check with Score-P and TopDown Plugin what is slow here
- [ ] Maybe integrate HDF5 deeply:
  - [ ] Use manual inlinining of the complete library
  - [ ] Make it a static library and use LTO


## Usefule comments
- Threading: One can use threading, but has to create a common `zfp_stream`, which can then be use using multiple bitstreams. https://github.com/LLNL/zfp/blob/a46fa8b91bf2d69f4ffcf04af4f908383828ba79/src/share/parallel.c#L12
- Non 4^d Chunks --> Use [partial blocks](https://zfp.readthedocs.io/en/release1.0.1/low-level-api.html#_CPPv4I0EN3zfp28encode_partial_block_stridedE6size_tP10zfp_streamPK6Scalar6size_t9ptrdiff_t)


---

# Cloning the repo
```
# git checkout -b zfp tags/b3901
git clone --branch zfp --recurse-submodules git@github.com:jdomke/llama.cpp.git
```

# Building/Installing dependencies
```
## requirements.txt
pip install -r requirements.txt
## zfp 1.0.1
pushd zfp && rm -rf build && mkdir build
pushd build && cmake .. -DBUILD_ZFORP=1 -DCMAKE_BUILD_TYPE=Release && make && popd
#pushd build && cmake .. -DBUILD_ZFORP=1 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS="-O0 -g" -DCMAKE_CXX_FLAGS="-O0 -g" && make && popd
popd
## openblas v0.3.28
pushd OpenBLAS && rm -rf build
make
make PREFIX=$(pwd)/build install
popd
```

# Building llama.cpp
```
## https://github.com/ggerganov/llama.cpp/blob/b3901/docs/build.md
export PKG_CONFIG_PATH="$(pwd)/OpenBLAS/build/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
rm -rf build/
#ZFPRATE<0 => perform reversible compression; >0 for chosen compresssion rate
ZFP="-DZFPDIM=4 -DZFPRATE=-1.0 -I$(pwd)/zfp/include/ -L$(pwd)/zfp/build/lib64/ -Wl,-rpath=$(pwd)/zfp/build/lib64/ -lzfp"
OPT="-O1 -g -DZFPDBG=1" #""
CBT="Debug"             #"Release"
cmake -B build \
  -DCMAKE_BUILD_TYPE="${CBT}" -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DGGML_GPROF=OFF -DGGML_NATIVE=ON -DGGML_LTO=ON \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DBLA_PREFER_PKGCONFIG=ON \
  -DCMAKE_C_FLAGS="${OPT} ${ZFP}" -DCMAKE_CXX_FLAGS="${OPT} ${ZFP}" \
#  -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=$(find /usr/local/cuda/ -name nvcc) -DCMAKE_CUDA_FLAGS="${OPT} ${ZFP} -allow-unsupported-compiler -ccbin gcc-13"
cmake --build build --config "${CBT}" --parallel $(nproc)
```

# Download Llama-3-8B in BF16 safetensor format
```
huggingface-cli \
  download meta-llama/Meta-Llama-3-8B --exclude "original/*" \
  --local-dir Meta-Llama-3-8B \
  --token <INSERT_YOUR_OWN>
```


# Convert to gguf
```
## https://www.reddit.com/r/LocalLLaMA/comments/18elm98/diy_converting_safetensors_format_to_gguf_on_a_mac/
python3 ./convert_hf_to_gguf.py Meta-Llama-3-8B \
  --outfile Meta-Llama-3-8B/Meta-Llama-3-8B-F32.gguf --outtype f32
```

# Create Importance Matrix (for some quantizations below)
```
## https://medium.com/@ingridwickstevens/quantization-of-llms-with-llama-cpp-9bbf59deda35 (search for 'Importance Matrix')
## (ONLY use imatrix when absolutely needed even so more might be able to use it see ggml.c#L21908)
wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip -O "$(dirname ${GG})/wiki.train.raw.zip"
unzip -p "$(dirname ${GG})/wiki.train.raw.zip" wikitext-2-raw/wiki.train.raw > "$(dirname ${GG})/wiki.train.raw"
## skip -ngl 300 if no gpu support or reduce layers if not enough mem (on A100 <2h)
./build/bin/llama-imatrix -m "${GG}-${I}.gguf" -f "$(dirname ${GG})/wiki.train.raw" \
  -o "${GG}-imatrix.dat" -t $(nproc) -ngl 300 \
  2>&1 | tee -a "${GG}.log"
### https://github.com/ggerganov/llama.cpp/blob/b3901/examples/perplexity/README.md#llama-3-8b-scoreboard (getting imatrix from there?!; llama-quantize logs show problems with downloaded imatrix -> try again to generate!?)
#wget https://huggingface.co/JohannesGaessler/llama.cpp_importance_matrices/resolve/main/imatrix-llama_3-8b-f16-10m_tokens.dat -O "${GG}-imatrix.dat"
```

# Quantize model
```
## https://huggingface.co/docs/hub/en/gguf#quantization-types
## https://medium.com/@qdrddr/the-easiest-way-to-convert-a-model-to-gguf-and-quantize-91016e97c987
GG="Meta-Llama-3-8B/Meta-Llama-3-8B"
I="F32"
for O in Q4_0 Q4_1 Q5_0 Q5_1 IQ2_M TQ1_0 TQ2_0 Q2_K IQ3_XXS IQ3_S IQ3_M Q3_K IQ3_XS Q3_K_S Q3_K_M Q3_K_L IQ4_NL IQ4_XS Q4_K Q4_K_S Q4_K_M Q5_K Q5_K_S Q5_K_M Q6_K Q8_0 Q4_0_4_4 Q4_0_4_8 Q4_0_8_8 F16 BF16 IQ1_S IQ1_M IQ2_S IQ2_XXS IQ2_XS Q2_K_S; do
  rm -f "${GG}-${O}.gguf"
  ./build/bin/llama-quantize --imatrix "${GG}-imatrix.dat" "${GG}-${I}.gguf" "${GG}-${O}.gguf" ${O} $(nproc) \
    2>&1 | tee -a "${GG}.log"
done
```

# Patch quantizer to add zfp
```
## starting point: https://github.com/ggerganov/llama.cpp/blob/b3901/src/llama.cpp#L19800
##  -> https://github.com/ggerganov/llama.cpp/blob/b3901/src/llama.cpp#L18681
##  -> https://github.com/ggerganov/llama.cpp/blob/b3901/ggml/src/ggml.c#L21884
##  ==> add code to: https://github.com/ggerganov/llama.cpp/blob/b3901/ggml/src/ggml.c#L21908
##      (suggest overwriting quantize_q8_0 or ggml_fp32_to_fp16_row (line L21936) if that makes inference code easy (no use of Kompute!?))
##
## see status:
git diff b3901..HEAD
```

# Patch inference ops (make sure its thread-safe for production version)
```
## overwrite traits https://github.com/ggerganov/llama.cpp/blob/b3901/ggml/src/ggml.c#L865
##  -> eg https://github.com/ggerganov/llama.cpp/blob/b3901/ggml/src/ggml-quants.c#L5518 but make sure it's called
##  ==> skip #if and go to naive impl https://github.com/ggerganov/llama.cpp/blob/b3901/ggml/src/ggml-quants.c#L5846
```

# Testing functionality
```
./build/bin/llama-cli \
    --model "${GG}-${O}.gguf" \
    --threads 1 --repeat_penalty 1.0 --prompt "Tell me a German joke. I've never heard one before." \
    --predict 50 --seed 1 --verbose
```

# Build and quantize zfp versions (as long as dim/rate for zfp are static we need multiple builds)
```
GG="Meta-Llama-3-8B/Meta-Llama-3-8B"
I="F32"; O="ZFP"
for ZFPDIM in 4 3 2 1; do
  for ZFPRATE in -1.0 16.0 12.0 8.0 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.5; do
    rm -f "${GG}-${O}_${ZFPDIM}_${ZFPRATE}.gguf"
    rm -rf "build_${ZFPDIM}_${ZFPRATE}"
    OPT=""; CBT="Release";
    ZFP="-DZFPDIM=${ZFPDIM} -DZFPRATE=${ZFPRATE} -I$(pwd)/zfp/include/ -L$(pwd)/zfp/build/lib64/ -Wl,-rpath=$(pwd)/zfp/build/lib64/ -lzfp"
    cmake -B "build_${ZFPDIM}_${ZFPRATE}" -DCMAKE_BUILD_TYPE="${CBT}" -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DGGML_GPROF=OFF -DGGML_NATIVE=ON -DGGML_LTO=ON \
      -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DBLA_PREFER_PKGCONFIG=ON -DCMAKE_C_FLAGS="${OPT} ${ZFP}" -DCMAKE_CXX_FLAGS="${OPT} ${ZFP}"
    cmake --build "build_${ZFPDIM}_${ZFPRATE}" --config "${CBT}" --parallel $(nproc)
    ./build_${ZFPDIM}_${ZFPRATE}/bin/llama-quantize "${GG}-${I}.gguf" "${GG}-${O}_${ZFPDIM}_${ZFPRATE}.gguf" ${O} $(nproc) 2>&1 | tee -a "${GG}.zfp.log"
  done
done
```

# Eval signal-to-noise or mean quare error (zfp vs fp32/input)
```
## compare in/out for llama_tensor_quantize_internal
##  -> in:f32_data_03 out:new_data_03 at https://github.com/ggerganov/llama.cpp/blob/b3901/src/llama.cpp#L18681
##  prototype:
/*TODO just debugging*/if (ZFPDBG && GGML_TYPE_ZFP == new_type){ // && ZFPRATE < 0) {
  const ggml_type_traits * qtype = ggml_get_type_traits(new_type);
  float *vali_array = (float*)malloc(nelements_matrix*sizeof(float));
  printf("befor dequan\n");fflush(stdout);
  qtype->to_float(new_data_03, vali_array, nelements_matrix);
  for(int i=0; i<10; i++) {
      printf("%f %f\n", f32_data[i],vali_array[i]);fflush(stdout);
  }
}
```

# Eval perplexity
```
## https://ai.meta.com/blog/meta-llama-3-1/
## https://github.com/ggerganov/llama.cpp/issues/8409
## https://github.com/ggerganov/llama.cpp/discussions/2321
## https://github.com/ggerganov/llama.cpp/blob/master/examples/perplexity/README.md
```

# Eval compression rate / model size [in byte]:
```
GG="Meta-Llama-3-8B/Meta-Llama-3-8B"
for O in Q4_0 Q4_1 Q5_0 Q5_1 IQ2_M TQ1_0 TQ2_0 Q2_K IQ3_XXS IQ3_S IQ3_M Q3_K IQ3_XS Q3_K_S Q3_K_M Q3_K_L IQ4_NL IQ4_XS Q4_K Q4_K_S Q4_K_M Q5_K Q5_K_S Q5_K_M Q6_K Q8_0 Q4_0_4_4 Q4_0_4_8 Q4_0_8_8 F16 BF16 IQ1_S IQ1_M IQ2_S IQ2_XXS IQ2_XS Q2_K_S  F32 ; do
  echo ${O} $(stat -c %s "${GG}-${O}.gguf")
done
### this would be incorrect, the current code writes max block size not the actually required block size
#for ZFPDIM in 4 3 2 1; do
#  for ZFPRATE in -1.0 16.0 12.0 8.0 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.5; do
#    O="ZFP"; echo ${O}_${ZFPDIM}_${ZFPRATE} $(stat -c %s "${GG}-${O}_${ZFPDIM}_${ZFPRATE}.gguf")
#  done
#done
```

# Eval bits/weight
```
## ???
```

# Eval ms/token
```
## 1) big prefill (1k+ words) + 1 output word
## 2) token split in 80% prefill + 20% prediction
## 3) small prefill/question + predicting hundred of words
```

# related work
```
## is SZx faster??? https://arxiv.org/pdf/2201.13020
## competing compressors https://sdrbench.github.io/
```

# old readme
[here](org.readme.md)
