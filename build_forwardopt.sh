#!/bin/bash

#### change this to your own LLVM INSTALL path
# LLVM_BUILD_DIR="/home/jhlou/LLVM17/llvm-project/build/"
# LLVM_INSTALL_DIR="/home/jhlou/LLVM17/llvm-project/build/"

LLVM_BUILD_DIR="/home/jhlou/LLVM17/llvm-project-4553dc46a05ec6f1e2aebcde1ce185772a26780b/build/"
LLVM_INSTALL_DIR="/home/jhlou/LLVM17/llvm-project-4553dc46a05ec6f1e2aebcde1ce185772a26780b/build/"
############
###  On 111 server
###  NOTE!!!!!!
###  change llvm path in top CMakeLists.txt!!!
module load gcc/9.4.0
mkdir build && cd build
cmake -GNinja \
  ..\
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \
  -DMLIR_DIR=$LLVM_INSTALL_DIR/lib/cmake/mlir \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON\
  -DCMAKE_INSTALL_PREFIX=. \
  -Dpybind11_DIR=/home/jhlou/Tools/pybind11-2.11.1/build/mock_install/share/cmake/pybind11

ninja -j 32 && ninja install

############
###  change llvm path in env.sh
source env.sh
# cmake --build . --target cgra-opt 
# cmake --build "$MY_BUILD_DIR" --target CGRA-opt soda-translate mlir-runner AllocaNamer XMLWriter SODAPythonModules VhlsLLVMRewriter
