//===- FORWARDOps.cpp - Operations of the FORWARD dialect -------------------------===//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir-c/Debug.h"
#include "FORWARD/Dialect/FORWARD/tosa_ext/tosa_ext.h"

#include "FORWARD/Support/Float16.h"
#include "FORWARD/Support/Module.h"
#include "FORWARD/Support/MathUtils.h"
#include "FORWARD/Support/ModuleInterpreter.h"
// #include <llvm/Support/Debug.h>
// #include "llvm/Support/CommandLine.h"
// #include "llvm/Support/SourceMgr.h"
// #include "llvm/Support/ToolOutputFile.h"
using namespace mlir::FORWARD;
namespace mlir {
namespace tosa {
class AddOp;

void tosa_add_infer(Operation* op, mlir::FORWARD::ModuleInterpreter* interpreter) {
  /// tensor
  llvm::errs() <<"tosa_add_shape_infer!!! \n"; 

  Operation* tensor_op0 = op->getOperand(0).getDefiningOp();
  if (tensor_op0) {
    llvm::outs() << "Defining Op: " << *tensor_op0 << "\n";
  } else {
    llvm::outs() << "The value has no defining operation.\n";
  }

  Operation* tensor_op1 = op->getOperand(1).getDefiningOp();
  if (tensor_op1) {
    llvm::outs() << "Defining Op: " << *tensor_op1 << "\n";
  } else {
    llvm::outs() << "The value has no defining operation.\n";
  }
  
  // std::shared_ptr<std::vector<float>> tensor = interpreter->mem_map[module::getName(tensor_op).str()];
  auto p = *(interpreter->inference_map[module::getName(op).str()]);

  // auto num_elem = module::getNumElements(op->getOutput());
  auto num_elem = interpreter->mem_map[module::getName(tensor_op0).str()]->size();
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = 0;
    // LLVM_DEBUG(llvm::outs() << "line " << i << ":");
    for (auto in : p.inputs) {
      if (in != nullptr) {
        // LLVM_DEBUG(llvm::outs() << in[i] << " ");
        p.outputs[0][i] += in[i];
      }
    }
    // LLVM_DEBUG(llvm::outs() << "\n");
  }

}


}
}