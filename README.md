<<<<<<< HEAD
# Forward-Opt
=======
forward-opt: MLIR project to optimize and deploy VIT forward inference on X86

Our project will be keep updating in：https://anonymous.4open.science/r/forward-0DC8/
=======================

Directories:

1. python: python tools for quantization.

2. tools: cpp of forward-opt

3. include/lib: c++ header/source for forwar-opt 

4. models: torch-mlir models to run

5. bash_tools: some bash scripts

6. bindings: library of pymlir/pybinding

## Build 

Change your llvm install path in build_forwardopt.sh and CMakeList.txt, and run build_forwardopt.sh (recommemd to run it through ctrl+C/ctrl+v line-by-line).

## Run

Change your llvm install path in env.sh, and source env.sh.

Then run the model transforming following scripts in ./models.(eg. ./models/simplr/scripts.sh). 

Pytorch to mlir needs torch-mlir python package.

### Dependencies
##### LLVM-17
Maybe we need a new version of llvm. Commit: 4553dc46a05ec6f1e2aebcde1ce185772a26780b

Please download it from

https://github.com/llvm/llvm-project/tree/4553dc46a05ec6f1e2aebcde1ce185772a26780b

Install it with ./bash_tools/build_llvm.sh

##### Pybind11

##### Newest torch-mlir 

Please download it from

https://github.com/segmentKOBE/torch-mlir-forward
>>>>>>> origin/new
