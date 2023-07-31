# This includes several libs that extends the linking time
# only the necessary libs are included instead
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# message(PUBLIC "${dialect_libs}")
# message(PUBLIC "${conversion_libs}")

set(LIBS
  # ${dialect_libs}
  # ${conversion_libs}

  # # Core
  # MLIRIR
  # MLIRTransforms
  # MLIROptLib
  
  # # Dialects
  # MLIRLLVMDialect
  # MLIRLinalgDialect
  # MLIRMemRefDialect
  # MLIRAffineDialect
  # MLIRArithDialect
  # MLIRMathDialect
  # MLIRFuncDialect
  # MLIRSCFDialect
  # MLIRMLProgramDialect
  # MLIRTensorDialect
  # MLIRTosaDialect
 
  # # Transforms
  # MLIRFuncTransforms
  # MLIRLinalgTransforms
  # MLIRLinalgTransformOps
  # MLIRAffineTransforms
  # MLIRAffineTransformOps
  # MLIRAffineAnalysis
  # MLIRSCFTransforms
  # MLIRReconcileUnrealizedCasts
  # MLIRMemRefTransforms

  # # Test Passes
  # MLIRLinalgTestPasses
  # MLIRAffineTransformsTestPasses
  
  # # Conversion
  # MLIRAffineToStandard
  # MLIRSCFToControlFlow
  # MLIRMemRefToLLVM
  # MLIRMathToLLVM
  # MLIRMathToLibm
  # MLIRArithToLLVM
  # MLIRFuncToLLVM
  # MLIRLinalgToLLVM
  # MLIRLinalgToStandard

  # # Forward
  # MLIRFORWARDMisc
  # MLIRFORWARDTransforms
  # MLIRFORWARDOps
  FORAWRDMLIRInitAll
  FORWARDMLIRSupport
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

set(SOURCES
  test-opt.cpp
)

add_llvm_executable(test-opt ${SOURCES})

llvm_update_compile_flags(test-opt)
target_link_libraries(test-opt PRIVATE ${LIBS})
llvm_update_compile_flags(test-opt)

mlir_check_link_libraries(test-opt)

install(TARGETS test-opt)