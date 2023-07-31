//===- soda-opt.cpp ---------------------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "FORWARD/Dialect/FORWARD/IR/FORWARD.h"
#include "FORWARD/Dialect/FORWARD/Transforms/Passes.h"
#include "FORWARD/Misc/Passes.h"

#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"

// Defined in the test directory, no public header.
namespace mlir {
void registerTestLoopPermutationPass();
namespace test {

int registerTestLinalgCodegenStrategy();
} // namespace test
} // namespace mlir

// Register important linalg passes
inline void registerLinalgPassesForSoda() {

  mlir::registerLinalgPasses();
}

// Register important affine passes
inline void registerAffinePassesForFORWARD() {

  mlir::affine::registerAffineDataCopyGenerationPass();
  mlir::affine::registerAffineLoopInvariantCodeMotionPass();
  mlir::affine::registerAffineLoopTilingPass();
  mlir::affine::registerAffineLoopFusionPass();
  mlir::affine::registerAffineLoopUnrollPass();
  mlir::affine::registerAffineScalarReplacementPass();
  mlir::affine::registerAffineLoopUnrollAndJamPass();
  mlir::affine::registerAffineLoopNormalizePass();
  mlir::affine::registerSimplifyAffineStructuresPass();

  // Test passes
  mlir::registerTestLoopPermutationPass();
}

int main(int argc, char **argv) {
  // mlir::registerAllDialects();
  // mlir::registerAllPasses();
  mlir::DialectRegistry registry;

  //===--------------------------------------------------------------------===//
  // Register mlir dialects and passes
  //===--------------------------------------------------------------------===//

  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();

  registerLinalgPassesForSoda();
  registerAffinePassesForFORWARD();
  mlir::bufferization::registerPromoteBuffersToStackPass();

  mlir::registerConvertLinalgToStandardPass();
  // mlir::registerConvertLinalgToLLVMPass(); // This pass maps linalg to blas
  mlir::registerLinalgLowerToAffineLoopsPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerSCFToControlFlowPass();
  mlir::registerConvertAffineToStandardPass();
  mlir::registerConvertMathToLLVMPass();
  mlir::registerConvertMathToLibmPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::arith::registerArithExpandOpsPass();
  mlir::memref::registerExpandOpsPass();
  mlir::memref::registerNormalizeMemRefsPass();
  mlir::registerReconcileUnrealizedCastsPass();

  // Add the following to selectively include the necessary dialects. You only
  // need to register dialects that will be *parsed* by the tool, not the one
  // generated
  // clang-format off
  registry.insert<mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect,
                  mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::vector::VectorDialect,
                  mlir::arith::ArithDialect,
                  mlir::affine::AffineDialect,
                  mlir::ml_program::MLProgramDialect,
                  mlir::tensor::TensorDialect,
                  mlir::FORWARD::FORWARDDialect,
                  mlir::tosa::TosaDialect>();
  // clang-format on
  // mlir::registerAllDialects(registry);

  // ----- My Dialect -----
  mlir::FORWARD::registerTestPrintOpNestingPass();
  mlir::registerImportCalibrationTable();
  mlir::registerTransformToQuantizedPass();


  return failed(
      mlir::MlirOptMain(argc, argv, "FORWARD optimizer driver\n", registry));
}
