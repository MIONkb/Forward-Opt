DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

### change this to your own llvm install path
# export LLVM_INSTALL_PATH="/home/jhlou/LLVM17/llvm-project/build"
export LLVM_INSTALL_PATH="/home/jhlou/LLVM17/llvm-project-4553dc46a05ec6f1e2aebcde1ce185772a26780b/build"


export PROJECT_ROOT=$DIR
export BUILD_PATH=${BUILD_PATH:-$PROJECT_ROOT/build}
export INSTALL_PATH=${INSTALL_PATH:-$PROJECT_ROOT/build}

echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "BUILD_PATH   : ${BUILD_PATH}"
echo "LLVM_INSTALL_PATH   : ${LLVM_INSTALL_PATH}"

export PATH=$INSTALL_PATH/bin:$PATH
# export PATH=$PROJECT_ROOT/llvm/bin:$PATH
#export PATH=$PROJECT_ROOT/python/tools:$PATH
export LD_LIBRARY_PATH=$INSTALL_PATH/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$INSTALL_PATH/bindings/pymlir:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$INSTALL_PATH/bindings/kld:$LD_LIBRARY_PATH

export PYTHONPATH=$PROJECT_ROOT/build/python:$PYTHONPATH
export PYTHONPATH=$LLVM_INSTALL_PATH/python_packages/mlir_core:$PYTHONPATH
# export PYTHONPATH=$LLVM_INSTALL_PATH/python_packages/mlir_core/mlir/_mlir_libs:$PYTHONPATH
# export PYTHONPATH=$PROJECT_ROOT/build/bindings:$PYTHONPATH
export PYTHONPATH=$LLVM_INSTALL_PATH/python:$PYTHONPATH
# export PYTHONPATH=$PROJECT_ROOT/third_party/caffe/python:$PYTHONPATH