//===- Test.h - Test dialect --------------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef CGRAOPT_DIALECT_FORWARD_IR_Test_H_
#define CGRAOPT_DIALECT_FORWARD_IR_Test_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

//===----------------------------------------------------------------------===//
// Test Dialect
//===----------------------------------------------------------------------===//

#include "FORWARD/Dialect/FORWARD/IR/FORWARDOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Test Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "FORWARD/Dialect/FORWARD/IR/FORWARDOps.h.inc"
#include "FORWARD/Dialect/FORWARD/IR/FORWARDOpsAttributes.h.inc"
#include "FORWARD/Dialect/FORWARD/IR/FORWARDOpsTypes.h.inc"
#include "FORWARD/Dialect/FORWARD/IR/FORWARDOpsEnums.h.inc"
//===----------------------------------------------------------------------===//
// FORWARD Dialect Helpers
//===----------------------------------------------------------------------===//

namespace mlir {
namespace FORWARD {

///////////////
/// Utilities.cpp
///////////////
// mlir::Operation* eraseKernel(::mlir::func::FuncOp& TopFunc, FDRA::KernelOp& Kernel);
// void SpecifiedAffineFortoKernel(::mlir::AffineForOp& forOp);
// AffineExpr getConstPartofAffineExpr(AffineExpr& expr);
// // void removeUnusedRegionArgs(Region &region);
// void eliminateUnusedIndices(Operation *op);
// SmallVector<Value> getOperandInRank(Operation *op, unsigned rank);

// ///// following 4 functions are defined to help extract kernel function
// // bool isSinkingBeneficiary(Operation *op);
// // static bool extractBeneficiaryOps(Operation *op, llvm::SetVector<Value> existingDependencies,
// //       llvm::SetVector<Operation *> &beneficiaryOps, llvm::SmallPtrSetImpl<Value> &availableValues);
// LogicalResult sinkOperationsIntoKernelOp(FDRA::KernelOp kernelOp);
// func::FuncOp GenKernelFunc(FDRA::KernelOp KernelOp, llvm::SetVector<Value> &operands);

//===----------------------------------------------------------------------===//
// A templated find func for smallvector
//===----------------------------------------------------------------------===//
template <typename T, unsigned N>
inline int findElement(const llvm::SmallVector<T, N>& vec, const T& elem) {
  for (unsigned i = 0; i < vec.size(); ++i) {
    if (vec[i] == elem) {
      return i;
    }
  }
  return -1;
}
} // namespace FORWARD
} // namespace mlir

#endif //CGRAOPT_DIALECT_FORWARD_IR_Test_H_
