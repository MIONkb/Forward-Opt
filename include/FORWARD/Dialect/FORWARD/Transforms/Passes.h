//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef FORWARD_DIALECT_PASSES_H_
#define FORWARD_DIALECT_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Quant/QuantOps.h"


namespace mlir {
class ModuleOp;
namespace FORWARD {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//
// Generate the code for registering passes.

std::unique_ptr<mlir::OperationPass<ModuleOp>> createImportCalibrationTablePass();
std::unique_ptr<mlir::OperationPass<ModuleOp>> createTransformToQuantizedPass();
std::unique_ptr<mlir::OperationPass<ModuleOp>> createMatmulToFunc();
} // namespace FORWARD

#define GEN_PASS_REGISTRATION
#include "FORWARD/Dialect/FORWARD/Transforms/Passes.h.inc"
} // namespace mlir

#endif // FORWARD_DIALECT_PASSES_H_
