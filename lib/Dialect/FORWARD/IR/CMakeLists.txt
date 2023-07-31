add_mlir_dialect_library(MLIRFORWARDOps
  FORWARDDialect.cpp
  FORWARDOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/FORWARD/Dialect/FORWARD

  DEPENDS
  #MLIRSNNOpInterfacesIncGen
  MLIRFORWARDOpInterfacesIncGen
  MLIRFORWARDOpsEnumsGen
  MLIRFORWARDOpsAttributesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRTensorDialect
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRFuncDialect
  MLIRViewLikeInterface
  )
