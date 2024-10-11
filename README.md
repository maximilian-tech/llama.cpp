# Cloning the repo
```
# git checkout -b zfp tags/b3901
git clone --branch zfp --recurse-submodules git@github.com:jdomke/llama.cpp.git
```

# Buildingi/Installing dependencies
```
## requirements.txt
pip install -r requirements.txt
## zfp
pushd zfp && rm -rf build && mkdir build
pushd build && cmake .. -DBUILD_ZFORP=1 -DCMAKE_BUILD_TYPE=Release && make && popd
popd
make -B
```

# Download Llama-3.1-70B in BF16 safetensor format
```
huggingface-cli \
  download meta-llama/Meta-Llama-3.1-70B --include "original/*" \
  --local-dir Meta-Llama-3.1-70B \
  --token <INSERT_YOUR_OWN>
```

# Convert to gguf
```
python ./convert_hf_to_gguf.py Meta-Llama-3.1-70B \
  --outfile Meta-Llama-3.1-70B/Meta-Llama-3.1-70B-fp32.gguf --outtype f32
```

# old readme
[here](org.readme.md)
