#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "FORWARD/Dialect/FORWARD/IR/FORWARD.h"
#include "FORWARD/Support/ModuleInterpreter.h"
namespace mlir {
namespace tosa {


class TransposeOp;

void tosa_transpose_shape_infer(Operation* op, mlir::FORWARD::ModuleInterpreter* interpreter);
void tosa_transpose_shape_infer(TransposeOp op, mlir::FORWARD::ModuleInterpreter* interpreter);


}
}