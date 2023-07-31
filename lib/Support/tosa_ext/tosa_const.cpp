//===- FORWARDOps.cpp - Operations of the FORWARD dialect -------------------------===//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir-c/Debug.h"
#include "FORWARD/Support/tosa_ext/tosa_const.h"

#include "FORWARD/Support/Float16.h"
#include "FORWARD/Support/Module.h"
#include "FORWARD/Support/MathUtils.h"
// #include <llvm/Support/Debug.h>
// #include "llvm/Support/CommandLine.h"
// #include "llvm/Support/SourceMgr.h"
// #include "llvm/Support/ToolOutputFile.h"

namespace mlir {
namespace tosa {
class ConstOp;

template <typename T> std::shared_ptr<std::vector<T>> read(ConstOp op) {
  auto type = op.getOutput().getType().cast<RankedTensorType>();
//   llvm::errs() <<" type:"<<type <<"\n";
  uint32_t store_mode = 0; //32bit check this
//   if (op.getStoreMode().has_value()) {
    // store_mode = StringSwitch<uint32_t>(op.getStoreModeAttr())
    //   .Case("1N", 0)
    //   .Case("2N", 1)
    //   .Case("4N", 2)
    //   .Default(0);
//   }
  std::string npz_loc = cast<StringAttr>(op.getOperation()->getAttr("npz_loc")).str();
  return module::weightFile().readTensor<T>(npz_loc, type, store_mode);
}

std::shared_ptr<std::vector<float>> read_as_float(ConstOp op){
  auto dtype = module::getStorageType(op.getOutput());
  if (dtype.isUnsignedInteger(8)) {
    auto data_u8 = read<uint8_t>(op);
    return std::make_shared<std::vector<float>>(data_u8->begin(),
                                                data_u8->end());
  } else if (dtype.isInteger(8)) {
    auto data_i8 = read<int8_t>(op);
    return std::make_shared<std::vector<float>>(data_i8->begin(),
                                                data_i8->end());
  } else if (dtype.isF32()) {
    // llvm::errs() << "[Info] In function read_as_float(constop), F32\n";
    return read<float>(op);
  } else if (dtype.isF16()) {
    auto data_u16 = read<uint16_t>(op);
    auto data_f32 = std::make_shared<std::vector<float>>(data_u16->size());
    for (uint64_t i = 0; i < data_u16->size(); i++) {
      data_f32->data()[i] = f16_to_f32(data_u16->data()[i]);
    }
    return data_f32;
  } else if (dtype.isBF16()) {
    auto data_u16 = read<uint16_t>(op);
    auto data_f32 = std::make_shared<std::vector<float>>(data_u16->size());
    for (uint64_t i = 0; i < data_u16->size(); i++) {
      data_f32->data()[i] = bf16_to_f32(data_u16->data()[i]);
    }
    return data_f32;
  } else if (dtype.isUnsignedInteger(16)) {
    auto data_u16 = read<uint16_t>(op);
    return std::make_shared<std::vector<float>>(data_u16->begin(),
                                                data_u16->end());
  } else if (dtype.isInteger(16)) {
    auto data_i16 = read<int16_t>(op);
    return std::make_shared<std::vector<float>>(data_i16->begin(),
                                                data_i16->end());
  } else if (dtype.isUnsignedInteger(32)) {
    auto data_u32 = read<uint32_t>(op);
    return std::make_shared<std::vector<float>>(data_u32->begin(),
                                                data_u32->end());
  } else if (dtype.isInteger(32)) {
    auto data_i32 = read<int32_t>(op);
    return std::make_shared<std::vector<float>>(data_i32->begin(),
                                                data_i32->end());
  }
  op.dump();
  llvm_unreachable("ConstOp data not support read as float now");
  return nullptr;
}

}
}