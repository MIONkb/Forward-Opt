//===-- Passes.td - FORWARD pass definition file --------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines the FORWARD Passes
//
//===----------------------------------------------------------------------===//
#ifndef FORWARD_DIALECT_PASSES
#define FORWARD_DIALECT_PASSES
include "FORWARD/Dialect/FORWARD/IR/FORWARDBase.td"
include "mlir/Pass/PassBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/Dialect/SCF/IR/SCFOps.td"
include "mlir/Dialect/Affine/IR/AffineOps.td"

def ImportCalibrationTable : Pass<"forward-import-calibration-table", "ModuleOp"> {
  let summary = "Import calibration table by tpuc-opt";
  let constructor = "::mlir::FORWARD::createImportCalibrationTablePass()";
  let options = [
    Option<"tableFile", "file", "std::string", /*default=*/"",
           "calibration table file path">,
    Option<"isAsymmetric", "asymmetric", "bool", /*default=*/"true",
           "true for asymmetric quantization, or false for symmetric">,
  ];
  let dependentDialects = ["linalg::LinalgDialect", "quant::QuantizationDialect"];
}


def TransformToQuantized : Pass<"forward-transform-to-quantized", "ModuleOp"> {
  let summary = "Convert tosa Ops to quantized tosa Ops";
  let constructor = "::mlir::FORWARD::createTransformToQuantizedPass()";
  let dependentDialects = ["mlir::tosa::TosaDialect"];
  let options = [
    Option<"mode", "mode", "std::string", /*default=*/"",
           "default quantization mode: INT8/BF16/F32">,
    Option<"qtable", "qtable", "std::string", /*default=*/"",
           "a table of Ops that quantized to specific mode">,
    Option<"isAsymmetric", "asymmetric", "bool", /*default=*/"false",
           "true for asymmetric quantization, or false for symmetric">,
    Option<"weightFileName", "weightFileName", "std::string", /*default=*/"",
           "weight file name to save">,
  ];
}


#endif // FORWARD_DIALECT_PASSES
