cmake_minimum_required(VERSION 2.8.12)

# if (POLICY CMP0048)
#   # cmake warns if loaded from a min-3.0-required parent dir, so silence the warning:
#   cmake_policy(SET CMP0048 NEW)
# endif()
set(BUILD_SHARED_LIBS ON)
project(pymlir)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\${ORIGIN}/")

find_package(pybind11 REQUIRED CONFIG)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

pybind11_add_module(pymlir pymlir.cpp)

llvm_update_compile_flags(pymlir)
target_compile_options(pymlir PRIVATE "-fexceptions")
target_compile_options(pymlir PRIVATE "-frtti")
target_link_libraries(pymlir PRIVATE
  FORAWRDMLIRInitAll
  FORWARDMLIRSupport
  # MLIRFORWARDTOSAExt
  # MLIRFORWARDOps

  # MLIRFuncDialect
  # MLIRFuncTransforms
  # MLIRFuncInlinerExtension

  MLIRTransforms
  MLIRParser
  MLIRCAPIDebug
  MLIRSupport
  MLIRIR
  LLVMCore
  LLVMSupport
)

llvm_update_compile_flags(pymlir)
mlir_check_link_libraries(pymlir)


install(TARGETS pymlir DESTINATION python)