//===-- FORWARD_BASE.td - Test dialect definitions ---------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines the FORWARD dialect
// 
//===----------------------------------------------------------------------===//

#ifndef FORWARD_BASE
#define FORWARD_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "FORWARD/Interfaces/InferenceInterface.td"

//===----------------------------------------------------------------------===//
// FORWARD Dialect.
//===----------------------------------------------------------------------===//

def FORWARD_Dialect : Dialect {
  let name = "FORWARD";
  let cppNamespace = "::mlir::FORWARD";

  let description = [{
    "a dialect for AI compiler-FORWARD"
  }];

  let dependentDialects = [
    // "scf::SCFDialect",
    // "::mlir::func"
  ];

  // let hasOperationAttrVerify = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // FORWARD_BASE
