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
popd
## openblas v0.3.28
pushd OpenBLAS && rm -rf build
make
make PREFIX=$(pwd)/build install
popd
```

# Building llama.cpp
```
## https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md
export PKG_CONFIG_PATH="$(pwd)/OpenBLAS/build/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
rm -rf build/
cmake -B build \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DGGML_GPROF=OFF -DGGML_NATIVE=ON -DGGML_LTO=ON \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DBLA_PREFER_PKGCONFIG=ON
cmake --build build --config Release --parallel $(nproc)
```

# Download Llama-3.1-70B in BF16 safetensor format
```
huggingface-cli \
  download meta-llama/Meta-Llama-3.1-70B --exclude "original/*" \
  --local-dir Meta-Llama-3.1-70B \
  --token <INSERT_YOUR_OWN>
```

# Convert to gguf
```
## https://www.reddit.com/r/LocalLLaMA/comments/18elm98/diy_converting_safetensors_format_to_gguf_on_a_mac/
python3 ./convert_hf_to_gguf.py Meta-Llama-3.1-70B \
  --outfile Meta-Llama-3.1-70B/Meta-Llama-3.1-70B-F32.gguf --outtype f32
```

# Quantize model
```
## https://medium.com/@qdrddr/the-easiest-way-to-convert-a-model-to-gguf-and-quantize-91016e97c987
GG="Meta-Llama-3.1-70B/Meta-Llama-3.1-70B"
IN="F32"
for OUT in Q4_0 Q4_1 Q5_0 Q5_1 IQ2_XXS IQ2_XS IQ2_S IQ2_M IQ1_S IQ1_M TQ1_0 TQ2_0 Q2_K Q2_K_S IQ3_XXS IQ3_S IQ3_M Q3_K IQ3_XS Q3_K_S Q3_K_M Q3_K_L IQ4_NL IQ4_XS Q4_K Q4_K_S Q4_K_M Q5_K Q5_K_S Q5_K_M Q6_K Q8_0 Q4_0_4_4 Q4_0_4_8 Q4_0_8_8 F16 BF16; do
  ./build/bin/llama-quantize "${GG}-${IN}.gguf" "${GG}-${OUT}.gguf" ${OUT} $(nproc)
done
```

# Eval
```
## https://ai.meta.com/blog/meta-llama-3-1/
## https://github.com/ggerganov/llama.cpp/issues/8409
## https://github.com/ggerganov/llama.cpp/discussions/2321
## https://github.com/ggerganov/llama.cpp/blob/master/examples/perplexity/README.md
```

# old readme
[here](org.readme.md)
