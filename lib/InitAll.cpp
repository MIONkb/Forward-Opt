//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "FORWARD/InitAll.h"
#include "FORWARD/Dialect/FORWARD/IR/FORWARD.h"
#include "FORWARD/Dialect/FORWARD/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir{
namespace FORWARD {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry
      .insert<mlir::tosa::TosaDialect, mlir::func::FuncDialect, 
              mlir::FORWARD::FORWARDDialect, mlir::quant::QuantizationDialect>();
}


void registerAllPasses() {
  registerCanonicalizer();
  // ::mlir::registerConversionPasses();
  ::mlir::registerFORWARDPasses();
  // tpu::registerTpuPasses();
}
} // namespace tpu_mlir
}
