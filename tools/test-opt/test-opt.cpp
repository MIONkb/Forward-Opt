//===- soda-opt.cpp ---------------------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

// #include "mlir/IR/Dialect.h"
// #include "mlir/IR/MLIRContext.h"
// #include "mlir/InitAllDialects.h"
// #include "mlir/InitAllPasses.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Pass/PassManager.h"
// #include "mlir/Support/FileUtilities.h"
// #include "mlir/Tools/mlir-opt/MlirOptMain.h"

// #include "mlir-c/Debug.h"

// #include "llvm/Support/CommandLine.h"
// #include "llvm/Support/InitLLVM.h"
// #include "llvm/Support/SourceMgr.h"
// #include "llvm/Support/ToolOutputFile.h"

// #include "FORWARD/Dialect/FORWARD/IR/FORWARD.h"
// #include "FORWARD/Dialect/FORWARD/Transforms/Passes.h"
// #include "FORWARD/Misc/Passes.h"

// #include "mlir/Dialect/Arith/Transforms/Passes.h"
// #include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir-c/Debug.h"
#include "mlir/Transforms/Passes.h"
#include "FORWARD/Dialect/FORWARD/IR/FORWARD.h"
#include "FORWARD/Support/ModuleInterpreter.h"
#include "FORWARD/InitAll.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

// Defined in the test directory, no public header.
// namespace mlir {
// void registerTestLoopPermutationPass();
// namespace test {

// int registerTestLinalgCodegenStrategy();
// } // namespace test
// } // namespace mlir

// // Register important linalg passes
// inline void registerLinalgPassesForSoda() {

//   mlir::registerLinalgPasses();
// }

// // Register important affine passes
// inline void registerAffinePassesForFORWARD() {

//   mlir::affine::registerAffineDataCopyGenerationPass();
//   mlir::affine::registerAffineLoopInvariantCodeMotionPass();
//   mlir::affine::registerAffineLoopTilingPass();
//   mlir::affine::registerAffineLoopFusionPass();
//   mlir::affine::registerAffineLoopUnrollPass();
//   mlir::affine::registerAffineScalarReplacementPass();
//   mlir::affine::registerAffineLoopUnrollAndJamPass();
//   mlir::affine::registerAffineLoopNormalizePass();
//   mlir::affine::registerSimplifyAffineStructuresPass();

//   // Test passes
//   mlir::registerTestLoopPermutationPass();
// }

int main(int argc, char **argv) {
  // mlir::registerAllDialects();
  // mlir::registerAllPasses();
  using namespace mlir;
  std::unique_ptr<mlir::MLIRContext> context_;
  context_.reset();
  OwningOpRef<ModuleOp> module_;
  std::string filename = "/home/jhlou/forward-opt/models/visionLinear/tosa_elided.mlir";
  mlirEnableGlobalDebug(true);  
  llvm::DebugFlag = true;
    
  DialectRegistry registry;

  registry.insert<func::FuncDialect, FORWARD::FORWARDDialect,
                    quant::QuantizationDialect, memref::MemRefDialect,
                    tensor::TensorDialect, tosa::TosaDialect>();
  context_ = std::make_unique<MLIRContext>(registry);

  module_ = parseSourceFile<mlir::ModuleOp>(filename, context_.get());
  std::cout << "file:" << filename<< ", module: "<< module_.get() <<"\n";
  module_.get().dump();

  
  // //===--------------------------------------------------------------------===//
  // // Register mlir dialects and passes
  // //===--------------------------------------------------------------------===//

  // mlir::registerInlinerPass();
  // mlir::registerCanonicalizerPass();
  // mlir::registerCSEPass();

  // registerLinalgPassesForSoda();
  // registerAffinePassesForFORWARD();
  // mlir::bufferization::registerPromoteBuffersToStackPass();

  // mlir::registerConvertLinalgToStandardPass();
  // // mlir::registerConvertLinalgToLLVMPass(); // This pass maps linalg to blas
  // mlir::registerLinalgLowerToAffineLoopsPass();
  // mlir::registerConvertFuncToLLVMPass();
  // mlir::registerFinalizeMemRefToLLVMConversionPass();
  // mlir::registerSCFToControlFlowPass();
  // mlir::registerConvertAffineToStandardPass();
  // mlir::registerConvertMathToLLVMPass();
  // mlir::registerConvertMathToLibmPass();
  // mlir::registerArithToLLVMConversionPass();
  // mlir::arith::registerArithExpandOpsPass();
  // mlir::memref::registerExpandOpsPass();
  // mlir::memref::registerNormalizeMemRefsPass();
  // mlir::registerReconcileUnrealizedCastsPass();

  // Add the following to selectively include the necessary dialects. You only
  // need to register dialects that will be *parsed* by the tool, not the one
  // generated
  // clang-format off
  // registry.insert<mlir::func::FuncDialect,
  //                 mlir::memref::MemRefDialect,
  //                 mlir::LLVM::LLVMDialect,
  //                 mlir::linalg::LinalgDialect,
  //                 mlir::math::MathDialect,
  //                 mlir::scf::SCFDialect,
  //                 mlir::cf::ControlFlowDialect,
  //                 mlir::vector::VectorDialect,
  //                 mlir::arith::ArithDialect,
  //                 mlir::affine::AffineDialect,
  //                 mlir::ml_program::MLProgramDialect,
  //                 mlir::tensor::TensorDialect,
  //                 mlir::FORWARD::FORWARDDialect,
  //                 mlir::tosa::TosaDialect>();
  // clang-format on
  // mlir::registerAllDialects(registry);

  //===--------------------------------------------------------------------===//
  // Register SODA dialects and passes
  //===--------------------------------------------------------------------===//

  // Dialects
  // registry.insert<mlir::soda::SODADialect>();
  // registry.insert<mlir::snn::SNNDialect>();
  // registry.insert<mlir::mytest::MyTestDialect>();
  // registry.insert<mlir::FDRA::FDRADialect>();
  // registry.insert<mlir::ComplexOP::ComplexOPDialect>();

  // ----- My Dialect -----
  // mlir::mytest::registerGetMACPass();
  // mlir::FDRA::registerFDRALoopCdfgGenPass();
  // mlir::FDRA::registerExtractAffineForToKernelPass();
  // mlir::FDRA::registerAdjustKernelMemoryFootprintPass();
  // mlir::FDRA::registerExtractKernelToFuncPass();
  // mlir::FDRA::registerAutoDesignSpaceExplorePass();
  // mlir::FORWARD::registerTestPrintOpNestingPass();
  // mlir::FDRA::registerConvertKernelCallToLLVMPass();
  // mlir::FDRA::registerHoistLoadStoreInLoopNestPass();

  // mlir::registerSCFForLoopCanonicalizationPass();
  
  // ----- SODA -----
  // Misc passes
  // mlir::soda::registerTestPrintOpNestingPass();
  // mlir::soda::registerTestArgumentsToXMLPass();
  // mlir::soda::registerEraseMemrefDeallocPass();
  // mlir::soda::registerForwardMemrefAllocPass();
  // mlir::soda::registerForwardLinalgFillPass();
  // mlir::soda::registerForwardMemrefCopyPass();

  // // SODA Passes
  // mlir::soda::registerSodaKernelOutliningPass();
  // mlir::soda::registerSodaKernelGenerationPass();
  // mlir::soda::registerSodaHostGenerationPass();
  // mlir::soda::registerSodaAsyncRegionPassPass();

  // // Outlining passes
  // mlir::soda::registerConvertAllToSODAPass();
  // mlir::soda::registerConvertOperationToSODAPass();
  // mlir::soda::registerConvertAffineForToSODAPass();
  // mlir::soda::registerConvertSCFForToSODAPass();
  // mlir::soda::registerConvertLinalgDotToSODAPass();
  // mlir::soda::registerConvertLinalgMatmulToSODAPass();
  // mlir::soda::registerConvertLinalgConvToSODAPass();
  // mlir::soda::registerConvertLinalgGenericToSODAPass();

  // // Optimization passes
  // mlir::soda::registerPassManagerMiscPass();
  // mlir::soda::registerSimpleLoweringPass();
  // mlir::soda::registerOptimizedForBambuPass();
  // mlir::soda::registerOptimizedForVitisHLSPass();

  // // Conversion passes

  // // ----- SNN -----
  // mlir::snn::registerSNNPrintPass();

  // return failed(
  //     mlir::MlirOptMain(argc, argv, "FORWARD optimizer driver\n", registry));
}


// : && /opt/GCC/GCC-9.4.0/bin/g++ -fPIC -fno-semantic-interposition -fvisibility-inlines-hidden 
// -Werror=date-time -Wall -Wextra -Wno-unused-parameter -Wwrite-strings -Wcast-qual 
// -Wno-missing-field-initializers -Wimplicit-fallthrough -Wno-class-memaccess -Wno-redundant-move -Wno-pessimizing-move -Wno-noexcept-type -Wdelete-non-virtual-dtor -Wsuggest-override -Wno-comment -Wno-misleading-indentation -fdiagnostics-color -g -Wl,-rpath-link,/home/jhlou/forward-opt/build/lib tools/forward-opt/CMakeFiles/forward-opt.dir/forward-opt.cpp.o -o bin/forward-opt -L/home/jhlou/LLVM17/llvm-project-4553dc46a05ec6f1e2aebcde1ce185772a26780b/build/lib   -L/home/jhlou/LLVM17/llvm-project/build/./lib -Wl,-rpath,"\$ORIGIN/../lib:/home/jhlou/LLVM17/llvm-project/build/./lib"  -lpthread  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRIR.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIROptLib.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLLVMDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLinalgDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRMemRefDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAffineDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRArithDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRMathDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRFuncDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRSCFDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRMLProgramDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRTensorDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRTosaDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRFuncTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLinalgTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAffineTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRSCFTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRReconcileUnrealizedCasts.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRMemRefTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLinalgTestPasses.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAffineTransformsTestPasses.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAffineToStandard.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRSCFToControlFlow.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRMemRefToLLVM.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRMathToLLVM.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRMathToLibm.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRArithToLLVM.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRFuncToLLVM.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLinalgToLLVM.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLinalgToStandard.a  lib/libMLIRFORWARDMisc.a  lib/libMLIRFORWARDTransforms.a  lib/libMLIRFORWARDOps.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRBytecodeWriter.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRDebug.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRQuantUtils.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRQuantDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLinalgTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRMemRefTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRFuncToLLVM.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRArithToLLVM.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRControlFlowToLLVM.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRTensorTilingInterfaceImpl.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLinalgUtils.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRTensorUtils.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRSCFTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRArithTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRFuncTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRTensorTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRGPUTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAsyncDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRExecutionEngineUtils.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMPasses.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMCoroutines.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMipo.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMVectorize.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMLinker.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMInstrumentation.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMCodeGen.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMIRPrinter.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMObjCARCOpts.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMTarget.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLLVMToLLVMIRTranslation.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAffineTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRSCFUtils.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRArithAttrToLLVMConversion.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRSCFToControlFlow.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRMemRefToLLVM.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRVectorToSCF.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRVectorToLLVM.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRVectorTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRBufferizationTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRVectorUtils.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRGPUOps.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRX86VectorTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRX86VectorDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRArmNeonDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRArmSVETransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRArmSVEDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAMXTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAMXDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRTargetLLVMIRExport.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRDLTIDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLLVMIRTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRNVVMDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRTranslateLib.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMFrontendOpenMP.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMScalarOpts.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMAggressiveInstCombine.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMInstCombine.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMTransformUtils.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLLVMCommonConversion.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLLVMDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMBitWriter.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMAnalysis.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMProfileData.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMSymbolize.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMDebugInfoPDB.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMDebugInfoMSF.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMDebugInfoDWARF.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMObject.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMIRReader.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMAsmParser.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMBitReader.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMMCParser.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMMC.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMDebugInfoCodeView.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMTextAPI.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLinalgDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRMathDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRParser.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRBytecodeReader.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAsmParser.a 
// /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRTilingInterface.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAffineToStandard.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRTransforms.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRCopyOpInterface.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRRuntimeVerifiableOpInterface.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAffineUtils.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRTransformUtils.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRRewrite.a  
// /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRPDLToPDLInterp.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRPDLInterpDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRPDLDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAffineAnalysis.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRSCFDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRBufferizationDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRSparseTensorDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRPresburger.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRVectorDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRVectorInterfaces.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRMaskableOpInterface.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRMaskingOpInterface.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRPass.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAnalysis.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRDataLayoutInterfaces.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRFuncDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRCallInterfaces.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRControlFlowDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRTensorDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRAffineDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRMemRefDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRControlFlowInterfaces.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRLoopLikeInterface.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRDestinationStyleOpInterface.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRShapedOpInterfaces.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRCastInterfaces.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRComplexDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRParallelCombiningOpInterface.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRSideEffectInterfaces.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMCore.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMBinaryFormat.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMTargetParser.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMRemarks.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMBitstreamReader.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRDialectUtils.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRArithUtils.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRArithDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRInferTypeOpInterface.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRInferIntRangeCommon.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRInferIntRangeInterface.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRDialect.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRViewLikeInterface.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRIR.a  /home/jhlou/LLVM17/llvm-project/build/lib/libMLIRSupport.a  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMSupport.a  -lrt  -ldl  -lpthread  -lm  /usr/lib/x86_64-linux-gnu/libz.so  /usr/lib/x86_64-linux-gnu/libtinfo.so  /home/jhlou/LLVM17/llvm-project/build/lib/libLLVMDemangle.a && :
// tools/forward-opt/CMakeFiles/forward-opt.dir/forward-opt.cpp.o: In function `main':
