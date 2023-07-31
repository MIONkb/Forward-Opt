# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror -Wno-pedantic -fno-strict-aliasing -Wno-maybe-uninitialized")
# file(GLOB _sources *.cpp Dnnl/*.cpp Helper/*.cpp)

add_llvm_library(FORWARDMLIRSupport
  MathUtils.cpp
  ModuleInterpreter.cpp
  Module.cpp
  Float16.cpp
  tosa_ext/tosa_const.cpp
  tosa_ext/tosa_transpose.cpp

  DEPENDS
  ModuleAttrGen
  # MLIROpsIncGen

  LINK_LIBS PUBLIC
  # MLIRFORWARDTOSAExt
  MLIRIR
  MLIRSupport
  MLIRQuantDialect
  MLIRTosaDialect
  MLIRQuantUtils
  MLIRInferenceInterface
  cnpy
  # dnnl
  # z
)

target_compile_options(FORWARDMLIRSupport PRIVATE "-fexceptions")