cmake_minimum_required(VERSION 3.13.4)
project(forward-opt LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "PATH: ${PATH}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
# set(RAAA_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(MLIRDetectPythonEnv)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

# set(LLVM_INCLUDE_DIRS "/home/jhlou/LLVM17/llvm-project/build/include")
# set(LLVM_LIBRARY_DIRS "/home/jhlou/LLVM17/llvm-project/build/lib")
set(LLVM_INCLUDE_DIRS "/home/jhlou/LLVM17/llvm-project-4553dc46a05ec6f1e2aebcde1ce185772a26780b/build/include")
set(LLVM_LIBRARY_DIRS "/home/jhlou/LLVM17/llvm-project-4553dc46a05ec6f1e2aebcde1ce185772a26780b/build/lib")

link_directories(${LLVM_LIBRARY_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(PROGRESSBAR_PATH ${PROJECT_SOURCE_DIR}/third_party/progressbar)
include_directories(${PROGRESSBAR_PATH}/include)

set(CNPY_PATH ${PROJECT_SOURCE_DIR}/third_party/cnpy)
include_directories(${CNPY_PATH})

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
ENDIF()

# if(MLIR_ENABLE_BINDINGS_PYTHON)
#   message(STATUS "Enabling SODA-OPT Python API")
#   add_subdirectory(python)
# else()
#   message(STATUS "SODA-OPT Python bindings are disabled.")
# endif()

add_subdirectory(include/FORWARD)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(python)
add_subdirectory(bindings)
add_subdirectory(third_party)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Werror -Wno-unused-result -Wreturn-type -Wunused-variable -fexceptions")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")