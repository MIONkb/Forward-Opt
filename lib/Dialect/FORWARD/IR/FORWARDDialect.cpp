//===- FORWARDDialect.cpp - MLIR Dialect for FORWARD Kernels implementation -------===//
//===----------------------------------------------------------------------===//
//
// This file implements the FORWARD kernel-related dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "FORWARD/Dialect/FORWARD/IR/FORWARD.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mlir::FORWARD;

#include "FORWARD/Dialect/FORWARD/IR/FORWARDOpsDialect.cpp.inc"

void FORWARDDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "FORWARD/Dialect/FORWARD/IR/FORWARDOps.cpp.inc"
      >();
}