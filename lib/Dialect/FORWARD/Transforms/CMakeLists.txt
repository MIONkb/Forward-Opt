add_mlir_dialect_library(MLIRFORWARDTransforms
  ImportCalibration.cpp
  TransformToQuantized.cpp


  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/FORWARD/Dialect/FORWARD/Transforms
  ${PROJ_INCLUDE_DIR}/FORWARD/Dialect/FORWARD/IR

  DEPENDS
  FORWARDPassIncGen

  LINK_LIBS PUBLIC
  FORWARDMLIRSupport
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSCFDialect
  MLIRAffineDialect
  MLIRAffineToStandard
  )

# set(LIBS
#   FORWARDPassIncGen
# )