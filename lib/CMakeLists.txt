add_subdirectory(Misc)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Interfaces)
# add_subdirectory(Conversion)
# add_subdirectory(CAPI)

add_mlir_library(FORAWRDMLIRInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRQuantDialect
  MLIRTosaDialect
  MLIRInferenceInterface
#   TPUMLIRTopToTpu
  MLIRFORWARDOps
#   TPUMLIRTopToTosa
)