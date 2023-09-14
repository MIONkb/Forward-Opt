//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "cnpy.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "FORWARD/Support/ModuleInterpreter.h"
#include "FORWARD/Dialect/FORWARD/IR/FORWARD.h"
#include "FORWARD/Support/MathUtils.h"
#include "FORWARD/Support/Module.h"
#include <llvm/Support/Debug.h>
#include "progressbar.hpp"
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>

<<<<<<< HEAD
#include "FORWARD/Support/tosa_ext/tosa_const.h"
#include "FORWARD/Support/tosa_ext/tosa_ext.h"
=======
#include "FORWARD/Dialect/FORWARD/tosa_ext/tosa_const.h"
#include "FORWARD/Dialect/FORWARD/tosa_ext/tosa_ext.h"
>>>>>>> d893dfaa1cd7d29ee553fcd73d9eb4f8287d1237

#define DEBUG_TYPE "interpreter"

static const int64_t MAX_COUNT_LIMIT = 0x100000000ll;
namespace mlir{
namespace FORWARD {
ModuleInterpreter::ModuleInterpreter(ModuleOp module) : module(module) {
  // module::init(module);
  // if (!module::isState(module::State::TOP_F32) &&
  //     !module::isState(module::State::TPU_LOWERED)) {
  //   llvm_unreachable("mlir state not support");
  // }
  mem_mode = mem_mode_t::ALL_TENSOR_IN_MEM;
  total_count = 0;
  for (auto func : module.getOps<::mlir::func::FuncOp>()) {
    // alloce buffer for all value
    func.walk([&](mlir::Operation* op) {
      // LLVM_DEBUG(llvm::dbgs() << "op: " << op << "\n");
      if(isInferenceOp(op)){
        for (auto r : op->getResults()) {
          total_count += module::getNumElements(r);
        }
      }
    });
  }
  LLVM_DEBUG(llvm::dbgs() << "Allocate size: "
                          << total_count * sizeof(float) / 1024 << " KB\n");
  if (total_count >= MAX_COUNT_LIMIT) {
    mem_mode = mem_mode_t::PART_TENSOR_IN_MEM;
  }
}

ModuleInterpreter::~ModuleInterpreter() {
  for (auto func : module.getOps<::mlir::func::FuncOp>()) {
    func.walk([&](Operation *op) {
      if (auto infer_op = llvm::dyn_cast<InferenceInterface>(op)) {
        auto name = module::getName(op).str();
        if (inference_map.find(name) != inference_map.end()) {
          infer_op.deinit(*inference_map[name]);
        }
      }
    });
  }
}

bool ModuleInterpreter::is_no_mem_op(Operation *op) {
  if (op == nullptr) {
    return false;
  }
  return isa<tosa::ReshapeOp>(op); 
}

void ModuleInterpreter::allocate_resources() {
  // module::setWeightFileName(weightnpz);
  llvm::errs() <<"weightnpz: " << weightnpz <<"\n";
  module::init(module, weightnpz);
  allocate_all_tensor_in_mem();
  // switch (mem_mode) {
  // case mem_mode_t::ALL_TENSOR_IN_MEM:
  //   allocate_all_tensor_in_mem();
  //   break;
  // case mem_mode_t::PART_TENSOR_IN_MEM:
  //   allocate_part_tensor_in_mem();
  //   break;
  // case mem_mode_t::ALL_TENSOR_IN_DISK:
  //   allocate_all_tensor_in_disk();
  //   break;
  // }
}

bool ModuleInterpreter::check_op_in_mem(Operation *op) {
  for (auto r : op->getResults()) {
    if (module::isNone(r)) {
      continue;
    } else {
      auto name = module::getName(r).str();
      if (mem_map.find(name) == mem_map.end()) {
        return false;
      }
    }
  }
  for (auto i : op->getOperands()) {
    if (module::isNone(i)) {
      continue;
    } else {
      auto name = module::getName(i).str();
      if (mem_map.find(name) == mem_map.end()) {
        return false;
      }
    }
  }
  return true;
}

void ModuleInterpreter::collect_tensor(Value v) {
  auto count = module::getNumElements(v);
  if (count == 0) {
    return;
  }
  std::string name;
  auto op = v.getDefiningOp();
  if(isInferenceOp(op)){
    // llvm::errs() << "op:" << op << "\n";
    name = cast<StringAttr>(op->getAttr("idx")).str();
  }
  else{
    name = module::getName(v).str();
    if (value_map.find(name) != value_map.end()) {
      return;
    }
  }

  mem_map[name] = std::make_shared<std::vector<float>>(count);
  value_map[name] = v;
  // name_map[&v] = name;
}

// void ModuleInterpreter::allocate_part_tensor_in_mem() {
//   all_tensor_names.clear();
//   value_map.clear();
//   mem_map.clear();
//   num_infer_op = 0;
//   int step = ceiling_func(total_count, MAX_COUNT_LIMIT);
//   int64_t idx = 0;
//   for (auto func : module.getOps<::mlir::func::FuncOp>()) {
//     // alloce buffer for all value
//     func.walk([&](Operation *op) {
//       if (op == func.getOperation() || isa<top::NoneOp>(op)) {
//         // self
//       } else if (isa<ReturnOp>(op)) {
//         for (auto v : op->getOperands()) {
//           auto name = module::getName(v).str();
//           output_names.push_back(name);
//           collect_tensor(v);
//           if (auto castOp = dyn_cast<tpu::CastOp>(v.getDefiningOp())) {
//             collect_tensor(castOp.getOutput());
//           }
//         }
//       } else if (auto in_op = dyn_cast<top::InputOp>(op)) {
//         auto v = in_op.getOutput();
//         collect_tensor(v);
//         auto name = module::getName(v).str();
//         input_names.push_back(name);
//       } else if (auto wOp = dyn_cast<top::WeightOp>(op)) {
//         auto v = wOp.getOutput();
//         auto name = module::getName(v).str();
//         value_map[name] = v;
//         mem_map[name] = wOp.read_as_float();
//         all_weight_names.push_back(name);
//       } else {
//         for (auto r : op->getResults()) {
//           auto num_users = std::distance(r.user_begin(), r.user_end());
//           if (num_users > 1) {
//             collect_tensor(r);
//           } else if (idx % (2 * step) < step) {
//             collect_tensor(r);
//           }
//         }
//         idx++;
//       }
//     });
//     module::detachWeightFile(); // to free weight memory
//     // input output buffers for ops
//     func.walk([&](InferenceInterface infer_op) {
//       num_infer_op++;
//       auto name = module::getName(infer_op).str();
//       // checkout in and out in memory
//       if (check_op_in_mem(infer_op)) {
//         auto param = std::make_shared<InferenceParameter>();
//         for (auto result : infer_op->getResults()) {
//           if (result.getType().isa<NoneType>()) {
//             param->outputs.push_back(nullptr);
//           } else {
//             auto o_name = module::getName(result).str();
//             param->outputs.push_back(mem_map[o_name]->data());
//           }
//         }
//         for (auto input : infer_op->getOperands()) {
//           if (module::isNone(input)) {
//             param->inputs.push_back(nullptr);
//           } else {
//             auto i_name = module::getName(input).str();
//             param->inputs.push_back(mem_map[i_name]->data());
//           }
//         }
//         LLVM_DEBUG(llvm::dbgs() << "init: '" << name << "'\n");
//         if (failed(infer_op.init(*param))) {
//           infer_op->dump();
//           llvm_unreachable("op inferece init failed");
//         }
//         inference_map[name] = param;
//       }
//     });
//   }
// }
// void ModuleInterpreter::allocate_all_tensor_in_disk() {
//   all_tensor_names.clear();
//   value_map.clear();
//   mem_map.clear();
//   num_infer_op = 0;
//   for (auto func : module.getOps<::mlir::func::FuncOp>()) {
//     // only weight, input and output save in memory
//     func.walk([&](Operation *op) {
//       if (op == func.getOperation() || isa<top::NoneOp>(op)) {
//         // self
//       } else if (isa<ReturnOp>(op)) {
//         for (auto v : op->getOperands()) {
//           auto name = module::getName(v).str();
//           output_names.push_back(name);
//           // only output in ddr. other tensors in disk
//           auto count = module::getNumElements(v);
//           mem_map[name] = std::make_shared<std::vector<float>>(count);
//           all_tensor_names.push_back(name);
//         }
//       } else {
//         for (auto result : op->getResults()) {
//           auto count = module::getNumElements(result);
//           if (count == 0) {
//             continue;
//           }
//           auto name = module::getName(result).str();
//           bool is_input = isa<top::InputOp>(op);
//           if (is_input) {
//             input_names.push_back(name);
//           }
//           value_map[name] = result;
//           if (auto wOp = llvm::dyn_cast<top::WeightOp>(op)) {
//             mem_map[name] = wOp.read_as_float();
//             all_weight_names.push_back(name);
//           } else if (is_input) {
//             mem_map[name] = std::make_shared<std::vector<float>>(count);
//             all_tensor_names.push_back(name);
//           }
//         }
//       }
//     });
//     module::detachWeightFile(); // to free weight memory
//   }
// }

void ModuleInterpreter::allocate_all_tensor_in_mem() {
  all_tensor_names.clear();
  value_map.clear();
  mem_map.clear();
  num_infer_op = 0;
  for (auto func : module.getOps<::mlir::func::FuncOp>()) {
    // alloce buffer for inputs of func
    Block &entryBlock = func.getBody().front();
    int idx = 0;
    for(BlockArgument arg : entryBlock.getArguments()){
      if(auto shapedType = arg.getType().dyn_cast<ShapedType>()){
        std::string name = "input_" + std::to_string(idx);
        int64_t count = shapedType.getNumElements();
        LLVM_DEBUG(llvm::dbgs() << " Function Input:" << name <<",count:" << count <<"\n" );
        mem_map[name] = std::make_shared<std::vector<float>>(count);
        value_map[name]=arg;
<<<<<<< HEAD
=======
        input_names.push_back(name);
        // name_map[&arg]=name;
>>>>>>> d893dfaa1cd7d29ee553fcd73d9eb4f8287d1237
        idx++;
      }
    }
    
    // alloce buffer for all value
    func.walk([&](Operation *op) {
      if (op == func.getOperation()) {
        // self
      } else if (isa<ReturnOp>(op)) {
        for (auto v : op->getOperands()) {
          collect_tensor(v);
          // auto name = cast<StringAttr>(v.getDefiningOp()->getAttr("idx")).str();
          auto name = module::getName(v).str();
          output_names.push_back(name);
        }
      } 
      else if (auto wOp = dyn_cast<tosa::ConstOp>(op)) {
        auto v = wOp.getOutput();
        std::string name = module::getName(v).str();
        // std::string npzloc = cast<StringAttr>(op->getAttr("npz_loc")).str();
        mem_map[name] = read_as_float(wOp);
        all_weight_names.push_back(name);
        value_map[name] = v;
<<<<<<< HEAD
      } 
      // else if (is_no_mem_op(op)) {
      //   LLVM_DEBUG(llvm::dbgs() << " Not ConstOp:" << *op <<"\n" );
      //   auto v = op->getResult(0);
      //   std::string name = module::getName(v).str();
      //   std::string in = module::getName(op->getOperand(0)).str();
      //   all_tensor_names.push_back(name);
      //   value_map[name] = v;
      // } 
=======
        // name_map[&v] = name;
      } 
>>>>>>> d893dfaa1cd7d29ee553fcd73d9eb4f8287d1237
      else {
        LLVM_DEBUG(llvm::dbgs() << " Not ConstOp:" << *op <<"\n" );
        all_tensor_names.push_back(module::getName(op).str());
        for (auto r : op->getResults()) {
          collect_tensor(r);
        }
      }
    });
    module::detachWeightFile(); // to free weight memory




    // input output buffers for all ops
    func.walk([&](Operation *op) {
<<<<<<< HEAD
=======
      // llvm::errs() << "opname: " << op->getName().getStringRef().str() << "\n";
>>>>>>> d893dfaa1cd7d29ee553fcd73d9eb4f8287d1237
      // op in function body
      if (isInferenceOp(op)){
      // if (auto infer_op = llvm::dyn_cast<InferenceInterface>(op)) {
        num_infer_op++;
        auto name = module::getName(op).str();
        // LLVM_DEBUG(llvm::dbgs() << "[Info] here op:" << name <<"\n" );
        auto param = std::make_shared<InferenceParameter>();
        for (auto result : op->getResults()) {
          if (result.getType().isa<NoneType>()) {
            param->outputs.push_back(nullptr);
          } else {
            auto o_name = module::getName(result).str();
            // LLVM_DEBUG(llvm::dbgs() << "[Info] o_name:" << o_name <<"\n" );
            param->outputs.push_back(mem_map[o_name]->data());
          }
        }
        // LLVM_DEBUG(llvm::dbgs() << "[Info] first for" <<"\n" );
        for (auto input : op->getOperands()) {
          // LLVM_DEBUG(llvm::dbgs() << "[Info] input" << input <<"\n" );
          if (module::isNone(input)) {
            param->inputs.push_back(nullptr);
            continue;
          }
          std::string input_name;
          if (BlockArgument::classof(input)) {
            // LLVM_DEBUG(llvm::dbgs() << "[Info] an input" << input <<"\n");
            for(auto iter:value_map){
              if(iter.second == input)
                input_name = iter.first;
            }
            // param->inputs.push_back(nullptr); check this
            // continue;
          }else{
            input_name = module::getName(input).str();
          }
          // LLVM_DEBUG(llvm::dbgs() << "[Info] input_name:" << input_name <<"\n");
          if (mem_map.find(input_name) == mem_map.end()) {
            input.dump();
            llvm_unreachable("input operands not allocated");
          } else {
            param->inputs.push_back(mem_map[input_name]->data());
          }
        }
        // LLVM_DEBUG(llvm::dbgs() << "[Info] second for" <<"\n");
        // LLVM_DEBUG(llvm::dbgs() << "init: '" << name << "'\n");
        // if (failed(infer_op.init(*param))) {
        //   op->dump();
        //   llvm_unreachable("op inferece init failed");
        // } Check this
        inference_map[name] = param;
      }
    });
  }
}

void ModuleInterpreter::fake_quant_weight() {
  module::init(module);
  LLVM_DEBUG(llvm::errs() << "start fake_quant_weight\n");
  std::vector<std::string> not_quant_weight_names;
  for (auto func : module.getOps<::mlir::func::FuncOp>()) {
    func.walk([&](Operation *op) {
      // if (isa<top::ConvOp>(op) || isa<top::MatMulOp>(op)) {
      //   auto bias_op = op->getOperands()[2].getDefiningOp();
      //   if (auto weight_op = dyn_cast<top::WeightOp>(bias_op)) {
      //     not_quant_weight_names.push_back(module::getName(bias_op).str());
      //   }
      // }
    });
  }

  for (auto &name : all_weight_names) {
    if (std::count(not_quant_weight_names.begin(), not_quant_weight_names.end(),
                   name)) {
      continue;
    }

    auto mem = *mem_map.at(name);
    auto max_value =
        std::max(std::abs(*std::max_element(mem.begin(), mem.end())),
                 std::abs(*std::min_element(mem.begin(), mem.end())));
    for (auto &data : mem) {
      data = std::round(data * 127 / max_value) * max_value / 127;
    }
  }
}

void ModuleInterpreter::invoke(bool express_type) {
  LLVM_DEBUG(llvm::errs() << "[Info] In function invoke().\n");
  switch (mem_mode) {
  case mem_mode_t::ALL_TENSOR_IN_MEM:
    invoke_all_in_mem(express_type);
    break;
  case mem_mode_t::PART_TENSOR_IN_MEM:
    invoke_part_in_mem(express_type);
    break;
  default:
    llvm_unreachable("Mem not enough, please use invoke_to_disk");
    break;
  }
}

void ModuleInterpreter::invoke_all_in_mem(bool express_type) {
  module::init(module);
  progressbar bar(num_infer_op);
  int flag = 0;
  std::string if_name;
  for (auto func : module.getOps<::mlir::func::FuncOp>()) {
    WalkResult result = func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<func::FuncOp>(*op)) {
        return WalkResult::advance();
      }
      std::string name;
      if (op->getLoc().isa<NameLoc>() || op->getLoc().isa<FusedLoc>()) {
        name = module::getName(op).str();
      }
      LLVM_DEBUG(llvm::dbgs() << "compute: '" << op << "'\n");
      if (flag && isa<func::FuncOp>(*(op->getParentOp()))) {
        flag = 0; // clear
      }

      /// check this
//       if (isa<tpu::IfOp, top::IfOp>(op)) {
//         std::optional<RegisteredOperationName> info =
//             op->getName().getRegisteredInfo();
//         if_name = name;
//         auto *inferInterface =
//             info->getInterface<FORWARD::InferenceInterface>();
//         if (failed(inferInterface->inference(inferInterface, op,
//                                              *inference_map[name]))) {
//           flag = 2; // else branch
//         } else {
//           flag = 1; // then branch
//         }
//         return WalkResult::advance();
//       } else if (isa<FORWARD::InferenceInterface>(op) && 0 == flag) {
//         bar.update();
//         auto infer_op = dyn_cast<InferenceInterface>(op);
//         if (failed(infer_op.inference(*inference_map[name]))) {
//           infer_op.dump();
//           llvm_unreachable("invoke failed!!");
//         }
//       } else if (flag && op->getParentRegion()->getRegionNumber() == flag - 1) {
//         if (auto infer_op = dyn_cast<InferenceInterface>(op)) {
//           if (failed(infer_op.inference(*inference_map[name]))) {
//             infer_op.dump();
//             llvm_unreachable("invoke failed!!");
//           }
//         }

//         if (isa<tpu::YieldOp, top::YieldOp>(op)) {
//           for (int k = 0; k < op->getNumOperands(); k++) {
//             auto num_element = module::getNumElements(op->getOperand(k));
//             name = module::getName(op->getOperand(k).getDefiningOp()).str();
// #pragma omp parallel for schedule(static, omp_schedule(num_element))
//             for (int i = 0; i < num_element; i++)
//               inference_map[if_name]->outputs[k][i] =
//                   inference_map[name]->outputs[k][i];
//           }
//         }
//       }

      return WalkResult::advance();
    });
  }
  llvm::errs() << "\n";
  // if (express_type && module::isState(module::State::TPU_LOWERED)) {
  //   for (auto &name : all_tensor_names) {
  //     auto value = value_map.at(name);
  //     if (is_no_mem_op(value.getDefiningOp())) {
  //       continue;
  //     }
  //     auto mem = mem_map.at(name);
  //     if (module::isUniformQuantized(value)) {
  //       auto qtype = module::getUniformQuantizedType(value);
  //       for (auto &data : *mem) {
  //         data = (data - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
  //       }
  //     }
  //   }
  // }
}

void ModuleInterpreter::value_to_disk(const std::string &filename,
                                      const std::string &name,
                                      std::vector<float> &data,
                                      bool express_type) {
  // auto value = value_map.at(name);
  // if (express_type && module::isState(module::State::TPU_LOWERED)) {
  //   if (module::isUniformQuantized(value)) {
  //     auto qtype = module::getUniformQuantizedType(value);
  //     for (auto &d : data) {
  //       d = (d - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
  //     }
  //   }
  // }
  // cnpy::npz_save(filename, name, data, "a");
  llvm_unreachable("Not Implemented");
}

void ModuleInterpreter::invoke_to_disk(const std::string &filename,
                                       bool express_type) {
  module::init(module);
  progressbar bar(num_infer_op);
  std::map<std::string, int> mem_uses;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](InferenceInterface infer_op) {
      bar.update();
      FORWARD::InferenceParameter p;
      std::vector<std::string> to_free;
      for (auto in : infer_op->getOperands()) {
        if (module::isNone(in)) {
          p.inputs.push_back(nullptr);
          continue;
        }
        auto name = module::getName(in).str();
        if (mem_map.find(name) == mem_map.end()) {
          in.dump();
          llvm_unreachable("input operands not allocated");
        } else {
          p.inputs.push_back(mem_map[name]->data());
        }
        auto iter = mem_uses.find(name);
        if (iter == mem_uses.end()) {
          continue;
        }
        iter->second--;
        if (iter->second == 0) {
          to_free.push_back(name);
        }
      }
      for (auto out : infer_op->getResults()) {
        if (module::isNone(out)) {
          p.outputs.push_back(nullptr);
          continue;
        }
        auto name = module::getName(out).str();
        auto mem_iter = mem_map.find(name);
        if (mem_iter != mem_map.end()) {
          p.outputs.push_back(mem_iter->second->data());
          continue;
        }
        auto count = module::getNumElements(out);
        auto mem = std::make_shared<std::vector<float>>(count);
        mem_map[name] = mem;
        p.outputs.push_back(mem->data());
        int num_uses = std::distance(out.user_begin(), out.user_end());
        mem_uses[name] = num_uses;
        if (num_uses == 0) {
          to_free.push_back(name);
        }
      }
      if (failed(infer_op.init(p))) {
        infer_op.dump();
        llvm_unreachable("init failed!!");
      }
      LLVM_DEBUG(llvm::dbgs() << "compute: '" << infer_op << "'\n");
      if (failed(infer_op.inference(p))) {
        infer_op.dump();
        llvm_unreachable("invoke failed!!");
      }
      for (auto &m : to_free) {
        value_to_disk(filename, m, *mem_map[m], express_type);
        mem_map.erase(m);
      }
      infer_op.deinit(p);
    });
  }
  llvm::errs() << "\n";
  for (auto &m : all_tensor_names) {
    value_to_disk(filename, m, *mem_map[m], express_type);
  }
}

void ModuleInterpreter::invoke_part_in_mem(bool express_type) {
  module::init(module);
  progressbar bar(num_infer_op);
  std::map<std::string, int> mem_uses;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](InferenceInterface infer_op) {
      bar.update();
      auto name = module::getName(infer_op).str();
      LLVM_DEBUG(llvm::dbgs() << "compute: '" << infer_op << "'\n");
      if (inference_map.find(name) != inference_map.end()) {
        if (failed(infer_op.inference(*inference_map[name]))) {
          infer_op.dump();
          llvm_unreachable("invoke failed!!");
        }
      } else {
        FORWARD::InferenceParameter p;
        std::vector<std::string> to_free;
        for (auto in : infer_op->getOperands()) {
          if (module::isNone(in)) {
            p.inputs.push_back(nullptr);
            continue;
          }
          auto name = module::getName(in).str();
          if (mem_map.find(name) == mem_map.end()) {
            in.dump();
            llvm_unreachable("input operands not allocated");
          } else {
            p.inputs.push_back(mem_map[name]->data());
          }
          auto iter = mem_uses.find(name);
          if (iter == mem_uses.end()) {
            continue;
          }
          iter->second--;
          if (iter->second == 0) {
            to_free.push_back(name);
          }
        }
        for (auto out : infer_op->getResults()) {
          if (module::isNone(out)) {
            p.outputs.push_back(nullptr);
            continue;
          }
          auto name = module::getName(out).str();
          auto mem_iter = mem_map.find(name);
          if (mem_iter != mem_map.end()) {
            p.outputs.push_back(mem_iter->second->data());
            continue;
          }
          auto count = module::getNumElements(out);
          auto mem = std::make_shared<std::vector<float>>(count);
          mem_map[name] = mem;
          p.outputs.push_back(mem->data());
          int num_uses = std::distance(out.user_begin(), out.user_end());
          mem_uses[name] = num_uses;
          if (num_uses == 0) {
            to_free.push_back(name);
          }
        }
        if (failed(infer_op.init(p))) {
          infer_op.dump();
          llvm_unreachable("init failed!!");
        }
        LLVM_DEBUG(llvm::dbgs() << "compute: '" << infer_op << "'\n");
        if (failed(infer_op.inference(p))) {
          infer_op.dump();
          llvm_unreachable("invoke failed!!");
        }
        for (auto &m : to_free) {
          mem_map.erase(m);
        }
        infer_op.deinit(p);
      }
    });
  }
  llvm::errs() << "\n";
  /// check this
  // if (express_type && module::isState(module::State::TPU_LOWERED)) {
  //   for (auto &name : all_tensor_names) {
  //     auto value = value_map.at(name);
  //     auto mem = mem_map.at(name);
  //     if (module::isUniformQuantized(value)) {
  //       auto qtype = module::getUniformQuantizedType(value);
  //       for (auto &data : *mem) {
  //         data = (data - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
  //       }
  //     }
  //   }
  // }
}

std::shared_ptr<std::vector<float>>
ModuleInterpreter::invoke_at(const std::string op_name) {
<<<<<<< HEAD
=======
  llvm::dbgs() << "[INFO] invoke at " << op_name <<"\n";
>>>>>>> d893dfaa1cd7d29ee553fcd73d9eb4f8287d1237
  //module::init(module);
  // LLVM_DEBUG(llvm::dbgs() << "[ERROR] print value_map when load!!!!!" <<"\n");
  // for(auto val : value_map){
  //   llvm::errs() << "[value_map]" << val.first << "; ";
  //   val.second.dump();
  // }
  if (value_map.find(op_name) == value_map.end()) {
    llvm::errs() << "Can't find op:" << op_name << "\n";
    llvm_unreachable("invoke_at op_name error");
  }
  auto v = value_map[op_name];
  auto op = v.getDefiningOp();
  LLVM_DEBUG(llvm::errs()<<"[invoke]op:" << op->getName().getStringRef().str());
<<<<<<< HEAD
  LLVM_DEBUG(llvm::errs()<<"[invoke]op:" << ::mlir::tosa::TransposeOp::getOperationName());
  if(op->getName().getStringRef().str() == ::mlir::tosa::TransposeOp::getOperationName()){
      ::mlir::tosa::tosa_transpose_shape_infer(op, this);
  }
=======
  LLVM_DEBUG(llvm::errs()<<"[invoke]operandNum:" << op->getNumOperands() << "\n");
  // op->dump();
  // this->module->dump();
  // LLVM_DEBUG(llvm::errs()<<"[invoke]op:" << ::mlir::tosa::TransposeOp::getOperationName());
  if(op->getName().getStringRef().str() == ::mlir::tosa::TransposeOp::getOperationName()){
      ::mlir::tosa::tosa_transpose_infer(op, this);
  }else if(op->getName().getStringRef().str() == ::mlir::tosa::AddOp::getOperationName()){
      ::mlir::tosa::tosa_add_infer(op, this);
  }else if(op->getName().getStringRef().str() == ::mlir::tosa::Conv2DOp::getOperationName()){
      ::mlir::tosa::tosa_conv2d_infer(op, this);
  }else if(op->getName().getStringRef().str() == ::mlir::tosa::MatMulOp::getOperationName()){
      ::mlir::tosa::tosa_matmul_infer(op, this);
  }
  // else if(op->getName().getStringRef().str() == ::mlir::tosa::TransposeOp::getOperationName()){
  //     ::mlir::tosa::tosa_transpose_shape_infer(op, this);
  // }
>>>>>>> d893dfaa1cd7d29ee553fcd73d9eb4f8287d1237
  // if (op == nullptr){
  // if (false == isa<InferenceInterface>(op)){
  //   llvm::errs() << "what's getDefiningOp???\n";
  // }
  // if (op == nullptr || false == isa<InferenceInterface>(op)) {
  //   llvm::errs() << "Op :" << op_name << " can't do inference\n";
  //   llvm_unreachable("invoke_at infer error");
  // }
  // auto infer_op = cast<InferenceInterface>(op);
  // LLVM_DEBUG(llvm::dbgs() << "invoke at: '" << infer_op << "'\n");
  // if (failed(infer_op.inference(*inference_map[op_name]))) {
  //   infer_op.dump();
  //   llvm_unreachable("infer_op.inference failed!!");
  // }

  return getTensor(op_name);
}

void ModuleInterpreter::invoke_from(const std::string op_name) {
  module::init(module);
  bool start_run = false;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](InferenceInterface infer_op) {
      auto name = module::getName(infer_op).str();
      if (name == op_name) {
        start_run = true;
      }
      LLVM_DEBUG(llvm::dbgs() << "invoke: '" << infer_op << "'\n");
      if (start_run && failed(infer_op.inference(*inference_map[name]))) {
        infer_op.dump();
        llvm_unreachable("invoke failed!!");
      }
    });
  }
}

// this function is specific for learning weight calibration, returns the
// gradent of weight in conv input is the grd of dst, and returns the gradent of
// weight
// void ModuleInterpreter::backward_weight_at(const std::string op_name,
//                                       const void *dst_grd, const int dst_grd_len, const void *weight_grd, const int weight_grd_len) {
//   module::init(module);
//   if (value_map.find(op_name) == value_map.end()) {
//     llvm::errs() << "Can't find op:" << op_name << "\n";
//     llvm_unreachable("invoke_at op_name error");
//   }
//   auto v = value_map[op_name];
//   auto op = v.getDefiningOp();
//   if (op == nullptr || !isa<top::ConvOp>(op)) {
//     llvm_unreachable("op.type not support backward_weight!!");
//   }

//   auto back_param = std::make_shared<InferenceParameter>();
//   for (auto result : op->getResults()) {
//     if (result.getType().isa<NoneType>()) {
//       continue;
//     }
//     auto type = result.getType().cast<RankedTensorType>();
//     auto name = module::getName(result).str();
//     size_t count = type.getNumElements();
//     if (count != dst_grd_len){
//       llvm_unreachable("output size mis-match");
//     }
//   }
//   back_param->inputs.push_back((float *)dst_grd);
//   if (auto convop = dyn_cast<top::ConvOp>(op)) {
//     auto opd = convop.getFilter();
//     if (opd.getType().isa<NoneType>()) {
//       llvm_unreachable("op.filter not exist!!");
//     }
//     auto type = opd.getType().cast<RankedTensorType>();
//     size_t count = type.getNumElements();
//     if (count != weight_grd_len){
//       llvm_unreachable("weight grd size mis-match!");
//     }
//   }
//   back_param->outputs.push_back((float*)weight_grd);

//   if (op == nullptr || false == isa<InferenceInterface>(op)) {
//     llvm::errs() << "Op :" << op_name << " can't do backward";
//     llvm_unreachable("backward weight error");
//   }
//   auto infer_op = cast<InferenceInterface>(op);
//   LLVM_DEBUG(llvm::dbgs() << "backward at: '" << op_name << "'\n");
//   if (failed(infer_op.backward_weight(*inference_map[op_name], *back_param))) {
//     infer_op.dump();
//     llvm_unreachable("infer_op.backward failed!!");
//   }
//   return;
// }
void ModuleInterpreter::printValuemap(){
<<<<<<< HEAD
  LLVM_DEBUG(llvm::dbgs() << "[ERROR] print value_map when load!!!!!" <<"\n");
=======
  llvm::dbgs() << "[ERROR] print value_map!!!!!" <<"\n";
>>>>>>> d893dfaa1cd7d29ee553fcd73d9eb4f8287d1237
  for(auto val : value_map){
    llvm::errs() << "[value_map]" << val.first << "; " << val.second  << "\n";
    // val.second.dump();
  }
<<<<<<< HEAD
=======
  llvm::dbgs() << "[ERROR] print end!!!!!" <<"\n";
}

void ModuleInterpreter::printMemmap(){
  llvm::dbgs() << "[ERROR] print mem_map!!!!!" <<"\n";
  for(auto mem : mem_map){
    llvm::errs() << "[mem_map]" << mem.first << "; " << mem.second->size()  << "\n";
    // val.second.dump();
  }
  llvm::dbgs() << "[ERROR] print end!!!!!" <<"\n";
>>>>>>> d893dfaa1cd7d29ee553fcd73d9eb4f8287d1237
}

void ModuleInterpreter::setTensor(const std::string &name, std::shared_ptr<std::vector<float>> data,
                                  size_t size, bool is_integer) {
  // module::init(module);
<<<<<<< HEAD
  for(auto mem : mem_map){
    llvm::errs() << "[mem_map]" << mem.first  << "\n";
  }
=======
  // for(auto mem : mem_map){
  //   llvm::errs() << "[mem_map]" << mem.first  << "\n";
  // }
>>>>>>> d893dfaa1cd7d29ee553fcd73d9eb4f8287d1237
  auto it = mem_map.find(name);
  if (it == mem_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, setTensor failed");
  }

  auto act = it->second;
  if (act->size() * sizeof( float) != size) {
    llvm::errs() << "Tensor " << name
                 << " data need size: " << act->size() * sizeof(float)
                 << " , but set size: " << size << "\n";
    llvm_unreachable("Error, setTensor failed");
  }
<<<<<<< HEAD
  memcpy(act->data(), data, size);
  // LLVM_DEBUG(llvm::dbgs() << "value" << value_map[name] << "\n");
  // auto value = value_map.at(name);
  // LLVM_DEBUG(llvm::dbgs() << "value" << value << "\n");
  // if (is_integer == false && module::isUniformQuantized(value)) {
  //     LLVM_DEBUG(llvm::dbgs() << "qtype \n");
  //   auto qtype = module::getUniformQuantizedType(value);
  //   float *p = (float *)data;
  //   for (uint32_t i = 0; i < act->size(); i++) {
  //     float d =
  //         p[i] * (float)(1 / qtype.getScale()) + (float)qtype.getZeroPoint();
  //     act->at(i) = qtype.isSigned() ? to_int8(d) : to_uint8(d);
  //   }
  // } else {
  //   LLVM_DEBUG(llvm::dbgs() << "before memcpy\n");
  //   memcpy(act->data(), data, size);
=======
  //获取原始指针
  // for(int i = 0; i < data->size(); i++){
  //   llvm::errs() << data->at(i) <<"\n";
  // }
  auto value = value_map.at(name);
  if (is_integer == false && module::isUniformQuantized(value)) {
    auto qtype = module::getUniformQuantizedType(value);
    for (uint32_t i = 0; i < act->size(); i++) {
      float d =
          data->at(i) * (float)(1 / qtype.getScale()) + (float)qtype.getZeroPoint();
      act->at(i) = qtype.isSigned() ? to_int8(d) : to_uint8(d);
    }
  }else{
    memcpy(act->data(), data->data(), size);
  }
  // const void* dataConstVoid = static_cast<const void*>(data.get());
  // memcpy(act->data(), dataConstVoid, size);
  // for(int i = 0; i < data->size(); i++){
  //   llvm::errs() << act->at(i) << " " << mem_map[name]->at(i) << " "<< data->at(i) <<"\n";
>>>>>>> d893dfaa1cd7d29ee553fcd73d9eb4f8287d1237
  // }
}

std::shared_ptr<std::vector<float>>
ModuleInterpreter::getTensor(const std::string &name, bool express_type) {
  auto it = mem_map.find(name);
  if (it == mem_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, getTensor failed");
  }

  // check this 
  // if (express_type && module::isState(module::State::TPU_LOWERED)) {
  //   auto value = value_map.at(name);
  //   if (module::isUniformQuantized(value)) {
  //     int i = 0;
  //     auto mem = mem_map.at(name);
  //     auto data_fp32 = std::make_shared<std::vector<float>>(it->second->size());
  //     auto qtype = module::getUniformQuantizedType(value);
  //     for (auto &data : *mem) {
  //       data_fp32->data()[i++] =
  //           (data - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
  //     }
  //     return std::move(data_fp32);
  //   }
  // }

  std::shared_ptr<std::vector<float>> tmp(it->second);
  return std::move(tmp);
}

bool ModuleInterpreter::getTensorQuantInfo(const std::string name,
                                           std::string &dtype, float &scale,
                                           int &zp) {
  auto it = mem_map.find(name);
  if (it == mem_map.end()) {
    return false;
  }
  auto value = value_map.at(name);
  auto stype = module::getStorageType(value);
  if (module::isUniformQuantized(value)) {
    auto qtype = module::getUniformQuantizedType(value);
    scale = qtype.getScale();
    zp = qtype.getZeroPoint();
    if (stype.isSignlessInteger(8) || stype.isUnsignedInteger(8))
      dtype = std::string("U8");
    else if (stype.isSignedInteger(8))
      dtype = std::string("I8");
    else if (stype.isSignlessInteger(16) || stype.isUnsignedInteger(16))
      dtype = std::string("U16");
    else if (stype.isSignedInteger(16))
      dtype = std::string("I16");
    else if (stype.isSignedInteger(32))
      dtype = std::string("I32");
    else if (stype.isSignlessInteger(32) || stype.isUnsignedInteger(32))
      dtype = std::string("U32");
    else {
      dtype = std::string("I4");
    }
  } else if (stype.isa<FloatType>()) {
    if (stype.isF16()) {
      dtype = std::string("F16");
      scale = 1.0;
      zp = 0;
    } else if (stype.isBF16()) {
      dtype = std::string("BF16");
      scale = 1.0;
      zp = 0;
    } else if (stype.isF32()) {
      dtype = std::string("F32");
      scale = 1.0;
      zp = 0;
    } else {
      dtype = std::string("NA");
      scale = 1.0;
      zp = 0;
    }
  } else if (stype.isa<IntegerType>()) {
    if (stype.isSignedInteger(8))
      dtype = std::string("I8");
    else if (stype.isSignlessInteger(8) || stype.isUnsignedInteger(8))
      dtype = std::string("I8"); // FIXME, seems fail to tell i8 from u8
    else if (stype.isSignlessInteger(16) || stype.isUnsignedInteger(16))
      dtype = std::string("U16");
    else if (stype.isSignedInteger(16))
      dtype = std::string("I16");
    else if (stype.isSignedInteger(32))
      dtype = std::string("I32");
    else if (stype.isSignlessInteger(32) || stype.isUnsignedInteger(32))
      dtype = std::string("U32");
    else {
      dtype = std::string("I4");
    }
    scale = 1.0;
    zp = 0;
  } else {
    dtype = std::string("UK");
    scale = 1.0;
    zp = 0;
  }
  return true;
}

llvm::ArrayRef<int64_t>
ModuleInterpreter::getTensorShape(const std::string &name) {
  auto it = value_map.find(name);
  if (it == value_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, getTensorShape failed");
  }
  return it->second.getType().cast<RankedTensorType>().getShape();
}



OpTable InferenceOpTable = 
{
  /// tosa dialect
  ::mlir::tosa::AddOp::getOperationName(),
  ::mlir::tosa::DivOp::getOperationName(),
  ::mlir::tosa::GreaterOp::getOperationName(),
  ::mlir::tosa::MulOp::getOperationName(),
  ::mlir::tosa::NegateOp::getOperationName(),
  ::mlir::tosa::PadOp::getOperationName(),
  ::mlir::tosa::PowOp::getOperationName(),
  ::mlir::tosa::ReciprocalOp::getOperationName(),
  ::mlir::tosa::AvgPool2dOp::getOperationName(),
  ::mlir::tosa::CeilOp::getOperationName(),
  ::mlir::tosa::ErfOp::getOperationName(),
  ::mlir::tosa::ExpOp::getOperationName(),
  ::mlir::tosa::LogOp::getOperationName(),
  ::mlir::tosa::RsqrtOp::getOperationName(),
  ::mlir::tosa::SigmoidOp::getOperationName(),
  ::mlir::tosa::SubOp::getOperationName(),
  ::mlir::tosa::TanhOp::getOperationName(),

  ::mlir::tosa::ReduceAllOp::getOperationName(),
  ::mlir::tosa::ReduceAnyOp::getOperationName(),
  ::mlir::tosa::ReduceMaxOp::getOperationName(),
  ::mlir::tosa::ReduceMinOp::getOperationName(),

  ::mlir::tosa::Conv2DOp::getOperationName(),
  ::mlir::tosa::Conv3DOp::getOperationName(),
  ::mlir::tosa::DepthwiseConv2DOp::getOperationName(),
  ::mlir::tosa::FFT2dOp::getOperationName(),
  ::mlir::tosa::FullyConnectedOp::getOperationName(),
  ::mlir::tosa::MatMulOp::getOperationName(),
  ::mlir::tosa::RFFT2dOp::getOperationName(),

  ::mlir::tosa::MaxPool2dOp::getOperationName(),
  ::mlir::tosa::MaximumOp::getOperationName(),
  ::mlir::tosa::MinimumOp::getOperationName(),
  ::mlir::tosa::TransposeConv2DOp::getOperationName(),
  ::mlir::tosa::TransposeOp::getOperationName(),
  ::mlir::tosa::ClampOp::getOperationName(),
  ::mlir::tosa::ReshapeOp::getOperationName(),
  // ::mlir::tosa::::getOperationName(),
  // ::mlir::tosa::::getOperationName(),
  // ::mlir::tosa::::getOperationName(),

};

bool isInferenceOp(mlir::Operation *op){
  return isInferenceOp(op->getName().getStringRef().str());
} 

bool isInferenceOp(const std::string &type){
  if(InferenceOpTable.count(type)){
    return true;
  }
  else 
    return false;
} 

} // namespace FORWARD
}
