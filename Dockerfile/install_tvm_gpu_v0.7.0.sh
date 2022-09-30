#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

apt-get update
apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
apt-get install -y git python3-pip lsb-release wget software-properties-common gnupg
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

cd /usr
git clone -b v0.7.0 https://github.com/apache/tvm --recursive
cd /usr/tvm

touch config.cmake
echo set\(USE_LLVM ON\) >> config.cmake
echo set\(USE_CUDA ON\) >> config.cmake
echo set\(USE_CUDNN ON\) >> config.cmake
echo set\(USE_BLAS openblas\) >> config.cmake
echo set\(USE_CUBLAS ON\) >> config.cmake
echo set\(USE_MPS ON\) >> config.cmake
mkdir build
cp config.cmake build

cd build
cmake ..
make -j4

apt-get install -y libprotobuf-dev protobuf-compiler
pip3 install --upgrade pip3
pip3 install --user numpy decorator attrs
pip3 install protobuf
pip3 install onnx