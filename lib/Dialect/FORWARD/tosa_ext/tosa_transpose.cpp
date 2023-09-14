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
class TransposeOp;

// int64_t getFLOPs() { return 0; }

// LogicalResult init(InferenceParameter &p) {
//   return success();
// }

// void deinit(InferenceParameter &p) {}

// LogicalResult inference(InferenceParameter &p) {
//   llvm_unreachable("Not Implemented");
//   return success();
// }
// void tosa_transpose_shape_infer(Operation* op, mlir::FORWARD::ModuleInterpreter* interpreter) {
//   tosa_transpose_shape_infer(cast<TransposeOp>(op) ,interpreter);
// }
// void tosa_transpose_shape_infer(TransposeOp op, mlir::FORWARD::ModuleInterpreter* interpreter) {
//   /// tensor
//   llvm::errs() <<"\ntosa_transpose_shape_infer!!! \n"; 
//   llvm::errs() << "operands num: "<<op.getNumOperands() << "\n";  
  
//   Operation* tensor_op = op.getOperand(0).getDefiningOp();
//   llvm::errs() <<"fp: "<< tensor_op ;   

//   std::shared_ptr<std::vector<float>> tensor = interpreter->mem_map[module::getName(tensor_op).str()];

//   for(auto fp : *tensor){
//     llvm::errs() <<"fp: "<< fp ;    
//   }

//   /// shape
//   Operation* shape = op.getOperation()->getOperand(1).getDefiningOp();
//   llvm::errs() <<"shape: "<< *shape;
// }

void tosa_transpose_infer(Operation* op, mlir::FORWARD::ModuleInterpreter* interpreter) {
  /// tensor
  llvm::errs() <<"tosa_transpose_shape_infer!!! \n"; 
  // llvm::errs() << "operands num: "<<op->getNumOperands() << "\n";   

  Operation* tensor_op1 = op->getOperand(1).getDefiningOp();
  if (tensor_op1) {
    llvm::outs() << "Defining Op: " << *tensor_op1 << "\n";
  } else {
    llvm::outs() << "The value has no defining operation.\n";
  }
  
  auto p = *(interpreter->inference_map[module::getName(op).str()]);

  // auto num_elem = module::getNumElements(op->getOutput());
  // auto num_elem = interpreter->mem_map[module::getName(tensor_op0).str()]->size();
  auto num_elem = module::getNumElements(op->getResults()[0]);
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = 0;
    // LLVM_DEBUG(llvm::outs() << "line " << i << ":");
    int inputidx = 0;
    for (auto in : p.inputs) {
      if (in != nullptr) {
        // LLVM_DEBUG(llvm::outs() << in[i] << " ");
      }
      if(inputidx == 0)
        p.outputs[0][i] = in[i];
      inputidx++;
    }
    // LLVM_DEBUG(llvm::outs() << "\n");
  }

  /// shape
  Operation* shape = op->getOperand(1).getDefiningOp();
  llvm::errs() <<"shape: "<< *shape;
  // auto dim0_ = getDim0();
  // auto dim1_ = getDim1();
  // auto in_shape = module::getShape(getInput());
  // auto num_dims = in_shape.size();
  // if (dim0_ < 0) {
  //   dim0_ += num_dims;
  // }
  // if (dim1_ < 0) {
  //   dim1_ += num_dims;
  // }
  // std::vector<int64_t> out_shape(in_shape);
  // if (in_shape.size() >= 2) {
  //   out_shape[dim0_] = in_shape[dim1_];
  //   out_shape[dim1_] = in_shape[dim0_];
  // }
  // module::setShapeOrVerify(getOutput(), out_shape);
  // std::vector<int64_t> order;
  // for (int i = 0; i < num_dims; ++i) {
  //   if (dim0_ == i) {
  //     order.push_back(dim1_);
  //   } else if (dim1_ == i) {
  //     order.push_back(dim0_);
  //   } else {
  //     order.push_back(i);
  //   }
  // }
  // auto op = getOperation();
  // OpBuilder builder(module::getCtx());
  // builder.setInsertionPointAfter(op);
  // // rewrite
  // std::vector<NamedAttribute> attrs;
  // attrs.push_back(
  //     builder.getNamedAttr("order", builder.getI64ArrayAttr(order)));
  // auto new_op = builder.create<PermuteOp>(getLoc(), getOutput().getType(),
  //                                         ValueRange{getInput()}, attrs);
  // op->replaceAllUsesWith(new_op);
}


}
}