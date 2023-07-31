//===- Passes.td - Pattern Match Optimizations for FORWARD -*- tablegen -*----===//
//===----------------------------------------------------------------------===//
//
// Defines language-specific pattern match optimizations for FDRA using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef FORWARD_MISC_PASSES
#define FORWARD_MISC_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Basic analysis passes
//===----------------------------------------------------------------------===//

def TestPrintOpNesting : Pass<"forward-test-print-nesting"> {
  let summary = "Print op nesting information";
  let description = [{
      Print nested intructure and metadata info associated with mlir operations.
  }];
  let constructor = "::mlir::FORWARD::createTestPrintOpNestingPass()";
  let options = [
    Option<"varName", "option-name", "uint64_t", /*default=*/"512",
           "Description of the option">,
  ];
}


//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite with Constraints
//===----------------------------------------------------------------------===//

#endif // FORWARD_MISC_PASSES
