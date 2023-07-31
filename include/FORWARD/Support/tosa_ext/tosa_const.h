#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir {
namespace tosa {
class ConstOp;

std::shared_ptr<std::vector<float>> read_as_float(ConstOp);

}
}