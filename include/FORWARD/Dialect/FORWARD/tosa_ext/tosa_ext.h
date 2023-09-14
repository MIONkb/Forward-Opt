#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "FORWARD/Dialect/FORWARD/IR/FORWARD.h"
#include "FORWARD/Support/ModuleInterpreter.h"

#include "FORWARD/Support/Module.h"
#include "FORWARD/Support/Dnnl/Dnnl.h"

namespace mlir {
namespace tosa {


// class TransposeOp;

void tosa_transpose_infer(Operation* op, mlir::FORWARD::ModuleInterpreter* interpreter);
// void tosa_transpose_shape_infer(TransposeOp op, mlir::FORWARD::ModuleInterpreter* interpreter);

// class AddOp;
void tosa_add_infer(Operation* op, mlir::FORWARD::ModuleInterpreter* interpreter);

// class ConvOp;
void tosa_conv2d_infer(Operation* op, mlir::FORWARD::ModuleInterpreter* interpreter);

void tosa_matmul_infer(Operation* op, mlir::FORWARD::ModuleInterpreter* interpreter);

}
}