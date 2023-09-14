add_mlir_dialect_library(MLIRFORWARDTOSAExt
  tosa_const.cpp
  tosa_transpose.cpp
  tosa_add.cpp
  tosa_conv2d.cpp
  tosa_matmul.cpp

  DEPENDS
  #MLIRSNNOpInterfacesIncGen
  MLIRFORWARDOpInterfacesIncGen
  MLIRFORWARDOpsEnumsGen
  MLIRFORWARDOpsAttributesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  FORWARDMLIRSupport
  MLIRTosaDialect
  MLIRDialectUtils
  MLIRIR
  MLIRTensorDialect
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRFuncDialect
  MLIRViewLikeInterface
  )
