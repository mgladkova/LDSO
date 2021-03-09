#!/usr/bin/env bash

BUILD_TYPE=Release
NUM_PROC=4

BASEDIR="$PWD"

cd "$BASEDIR/thirdparty/DBoW3"
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j$NUM_PROC

cd "$BASEDIR/thirdparty/g2o"
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j$NUM_PROC

cd "$BASEDIR/thirdparty/"
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
rm libtorch-cxx11-abi-shared-with-deps-latest.zip

cd "$BASEDIR"
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_PREFIX_PATH=$BASEDIR/thirdparty/libtorch ..
make -j$NUM_PROC
