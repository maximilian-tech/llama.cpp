# Cloning the repo
```
# git checkout -b zfp tags/b3901
git clone --branch zfp --recurse-submodules git@github.com:jdomke/llama.cpp.git
```

# Building zfp dependency
```
pushd zfp && rm -rf build && mkdir build
pushd build && cmake .. -DBUILD_ZFORP=1 -DCMAKE_BUILD_TYPE=Release && make && popd
popd
make -B
```

# 

# old readme
[here](org.readme.md)
