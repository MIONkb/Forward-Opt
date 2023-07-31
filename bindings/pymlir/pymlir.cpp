//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

// -------------
// pure C++ code
// -------------
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir-c/Debug.h"
#include "mlir/Transforms/Passes.h"
#include "FORWARD/Dialect/FORWARD/IR/FORWARD.h"
#include "FORWARD/Support/ModuleInterpreter.h"
#include "FORWARD/InitAll.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace mlir::FORWARD;

typedef std::map<std::string, std::shared_ptr<std::vector<float>>> tensor_map_t;
typedef std::map<std::string, std::vector<int64_t>> shape_map_t;

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// Warning: buffer in C++. New inference will erase old output
static py::array getPyArray(std::shared_ptr<std::vector<float>> ptr,
                            const std::vector<int64_t> &shape) {
  auto shared_ptr_ptr = new std::shared_ptr<std::vector<float>>(std::move(ptr));
  py::capsule delete_shared_ptr_ptr(shared_ptr_ptr, [](void *ptr) {
    delete reinterpret_cast<std::shared_ptr<std::vector<float>> *>(ptr);
  });
  return py::array_t<float>(shape, (*shared_ptr_ptr)->data(),
                            delete_shared_ptr_ptr);
}

struct quant_brief_info {
  std::string dtype;
  std::string shape;
  float scale;
  int zp;
};

class py_module {
public:
  py_module(){}
  py_module(std::string filename) {
    mlirEnableGlobalDebug(true);  
    llvm::DebugFlag = true;
    
    DialectRegistry registry;

    registry.insert<func::FuncDialect, FORWARD::FORWARDDialect,
                    quant::QuantizationDialect, memref::MemRefDialect,
                    tensor::TensorDialect, tosa::TosaDialect>();
    context_ = std::make_unique<MLIRContext>(registry);

    OwningOpRef<ModuleOp> module_OOR;
    module_OOR = parseSourceFile<mlir::ModuleOp>(filename, context_.get());
    module_ = module_OOR.get();

    std::cout << "file:" << filename<< ", module: "<< module_ <<"\n";
    assert(module_);
  }
  py_module(py::capsule capsuleOBJ) {
    // void * moduleOpPtr = PyCapsule_GetPointer(capsuleOBJ, "mlir.ir.Module._CAPIPtr");
    void * moduleOpPtr = capsuleOBJ.get_pointer();

    // mlir::ModuleOp* moduleOp = reinterpret_cast<mlir::ModuleOp*>(moduleOpPtr);
    // mlirEnableGlobalDebug(true);  
    // llvm::DebugFlag = true;
    
    // DialectRegistry registry;

    // registry.insert<func::FuncDialect, FORWARD::FORWARDDialect,
    //                 quant::QuantizationDialect, memref::MemRefDialect,
    //                 tensor::TensorDialect, tosa::TosaDialect>();
    // context_ = std::make_unique<MLIRContext>(registry);

    // OwningOpRef<ModuleOp> module_OOR;
    // module_OOR = parseSourceFile<mlir::ModuleOp>(filename, context_.get());
    // module_ = moduleop;

    // std::cout << " module: "<< *moduleOp <<"\n";
    assert(module_);
  }
  ~py_module() {
    interpreter_.reset();
    // auto module = module_.release();
    // if (module) {
    //   module.erase();
    // }
    // context_.reset();
  }

  void load(std::string filename) {
    if (context_) {
      context_.reset();
    }
    // registry.insert<func::FuncDialect, top::TopDialect, tpu::TpuDialect,
    //                 quant::QuantizationDialect>();

    mlirEnableGlobalDebug(true);  
    llvm::DebugFlag = true;
    
    DialectRegistry registry;

    registry.insert<func::FuncDialect, FORWARD::FORWARDDialect,
                    quant::QuantizationDialect, memref::MemRefDialect,
                    tensor::TensorDialect, tosa::TosaDialect>();
    context_ = std::make_unique<MLIRContext>(registry);

    OwningOpRef<ModuleOp> module_OOR;
    module_OOR = parseSourceFile<mlir::ModuleOp>(filename, context_.get());
    module_ = module_OOR.get();

    LLVM_DEBUG(llvm::dbgs() << "file:" << filename<< ", module: "<< module_ <<"\n");
    assert(module_);

    if (interpreter_) {
      interpreter_.reset();
    }

    interpreter_ = std::make_unique<FORWARD::ModuleInterpreter>(module_);
    if(weightFilePath_!=""){
      interpreter_->weightnpz = weightFilePath_;
      // interpreter_->setweightnpz(path);
    }
    interpreter_->allocate_resources();
    for (auto &name : interpreter_->input_names) {
      input_names.append(name);
    }
    for (auto &name : interpreter_->output_names) {
      output_names.append(name);
    }
    for (auto &name : interpreter_->all_tensor_names) {
      all_tensor_names.append(name);
    }
    for (auto &name : interpreter_->all_weight_names) {
      all_weight_names.append(name);
    }


    /////////
    /// get inputs
    /////////
    // func::FuncOp main_func;
    // auto funcOps = moduleOp.getOps<mlir::FuncOp>();
    // main_func = funcOps[funcOps.size()-1];
    // mlir::BlockArgumentRange args = ;
    // for(auto arg : main_func.getArguments()){

    // }
    // LLVM_DEBUG(llvm::dbgs() << "main_func:" << main_func<<"\n");

    // list(self.main_func.arguments)
    LLVM_DEBUG(llvm::dbgs() << "[Info] Load of py_module finished!\n");
  }

  py::dict getAllTensor() {
    py::dict py_ret;
    for (auto &name : interpreter_->all_tensor_names) {
      auto tensor = interpreter_->getTensor(name);
      auto shape = interpreter_->getTensorShape(name);
      py::str py_s(name);
      py_ret[py_s] = getPyArray(std::move(tensor), shape);
    }
    return py_ret;
  }

  void set_tensor(
      std::string name,
      py::array_t<float, py::array::c_style | py::array::forcecast> data) {

    LLVM_DEBUG(llvm::dbgs() << "[Info] In function set_tensor," << " name:"<< name <<"\n");
    interpreter_->setTensor(name, data.data(), data.size() * sizeof(float),
                            false);
  }

  void set_tensor_from_int(
      std::string name,
      py::array_t<float, py::array::c_style | py::array::forcecast> data) {
    interpreter_->setTensor(name, data.data(), data.size() * sizeof(float),
                            true);
  }

  // Warning: using copy in python
  py::array get_tensor(std::string name) {
    auto tensor = interpreter_->getTensor(name);
    auto shape = interpreter_->getTensorShape(name);
    return getPyArray(std::move(tensor), shape);
  }

  // Tip: not using copy in python, since independent mem
  py::array get_fp32_tensor(std::string name) {
    auto tensor = interpreter_->getTensor(name, true);
    auto shape = interpreter_->getTensorShape(name);
    return getPyArray(std::move(tensor), shape);
  }

  struct quant_brief_info format_tensor_qinfo(std::string name) {
    struct quant_brief_info q_info;
    if (!interpreter_->getTensorQuantInfo(name, q_info.dtype, q_info.scale,
                                          q_info.zp)) {
      q_info.dtype = std::string("NA");
      q_info.scale = 1.0;
      q_info.zp = 0;
      q_info.shape = std::string("[]");
      return q_info;
    }
    std::vector<int64_t> shape_ = interpreter_->getTensorShape(name);
    q_info.shape = std::string("[");
    for (int i = 0; i < shape_.size(); i++) {
      q_info.shape += std::to_string(shape_[i]);
      if (i != shape_.size() - 1)
        q_info.shape += std::string(", ");
    }
    q_info.shape += std::string("]");
    return q_info;
  }

  void invoke() { interpreter_->invoke(); }
  void fake_quant_weight() { interpreter_->fake_quant_weight(); }

  py::array invoke_at(const std::string name) {
    auto tensor = interpreter_->invoke_at(name);
    auto shape = interpreter_->getTensorShape(name);
    return getPyArray(std::move(tensor), shape);
  }

  // py::array backward_weight_at(
  //     const std::string name, const std::string weight_name,
  //     py::array_t<float, py::array::c_style | py::array::forcecast> grd_dst) {
  //   auto shape = interpreter_->getTensorShape(weight_name);
  //   size_t size = 1;
  //   for (auto dim : shape) {
  //     size *= dim;
  //   }
  //   py::array_t<float, py::array::c_style | py::array::forcecast> weight_grd(
  //       shape);
  //   interpreter_->backward_weight_at(name, grd_dst.data(), grd_dst.size(),
  //                                    weight_grd.data(), size);
  //   return weight_grd;
  // }

  void invoke_from(const std::string name) { interpreter_->invoke_from(name); }

  // void _set_context_(std::unique_ptr<mlir::MLIRContext> contextptr){context = *contextptr;}
  void _set_module_(ModuleOp md){module_ = md;}
  void _set_interpreter_(std::unique_ptr<FORWARD::ModuleInterpreter>&& interpreter){ interpreter_ = std::move(interpreter);}
  void set_weightFilePath_(std::string path){ 

    weightFilePath_ = path;
    LLVM_DEBUG(llvm::dbgs() << "weightFilePath_:" << weightFilePath_ <<"\n");
    // interpreter_->weightnpz = path;
  }
  // ModuleOp get_module(){ return module_.get();}
public:
  py::list all_tensor_names;
  py::list all_weight_names;
  py::list input_names;
  py::list output_names;
  py::list inputs;
  static std::string version;

private:
  std::unique_ptr<mlir::MLIRContext> context_;
  // mlir::MLIRContext context;
  // OwningOpRef<ModuleOp> module_;
  ModuleOp module_;
  std::string weightFilePath_;
  std::unique_ptr<FORWARD::ModuleInterpreter> interpreter_;
  // FORWARD::ModuleInterpreter interpreter_;
};

void debug_only(std::vector<std::string> debug_types) {
  llvm::DebugFlag = true;
  std::vector<const char *> c_debug;
  c_debug.reserve(debug_types.size());

  for (auto &d : debug_types)
    c_debug.push_back(const_cast<char *>(d.c_str()));
  llvm::setCurrentDebugTypes(c_debug.data(), c_debug.size());
}

void debug(bool enable) { llvm::DebugFlag = enable; }

#ifndef MLIR_VERSION
#define MLIR_VERSION "version unknown"
#endif

std::string py_module::version = MLIR_VERSION;

int int_add(int i, int j){
  return i + j;
}

py::object load_py_module(std::string filename) {
  // mlir::registerAllDialects();
  // mlir::registerAllPasses();
  std::cout << "load_py_module()"  <<"\n";
  using namespace mlir;
  std::unique_ptr<mlir::MLIRContext> context_;
  context_.reset();
  OwningOpRef<ModuleOp> module_;
  // std::string filename = "/home/jhlou/forward-opt/models/visionLinear/tosa_elided.mlir";
  mlirEnableGlobalDebug(true);  
  llvm::DebugFlag = true;
    
  DialectRegistry registry;

  registry.insert<func::FuncDialect, FORWARD::FORWARDDialect,
                    quant::QuantizationDialect, memref::MemRefDialect,
                    tensor::TensorDialect, tosa::TosaDialect>();
  context_ = std::make_unique<MLIRContext>(registry);

  module_ = parseSourceFile<mlir::ModuleOp>(filename, context_.get());
  std::cout << "file:" << filename<< ", module: "<< module_.get() <<"\n";
  module_.get().dump();

  // std::unique_ptr<FORWARD::ModuleInterpreter> interpreter_;
  // interpreter_  = std::make_unique<FORWARD::ModuleInterpreter>(module_.get());


  // interpreter_->allocate_resources();
  py_module md;

  // md._set_context_(context_);
  md._set_module_(module_.get());
  // md._set_interpreter_(std::move(interpreter_));

  // for (auto &name : interpreter_->input_names) {
  //   md.input_names.append(name);
  // }
  // for (auto &name : interpreter_->output_names) {
  //   md.output_names.append(name);
  // }
  // for (auto &name : interpreter_->all_tensor_names) {
  //   md.all_tensor_names.append(name);
  // }
  // for (auto &name : interpreter_->all_weight_names) {
  //   md.all_weight_names.append(name);
  // }

  return py::cast(std::move(md));
}

// wrap as Python module
PYBIND11_MODULE(pymlir, m) {
  m.doc() = "pybind11 for mlir";
  m.def("debug", &debug, py::arg("enable") = true,
        "enable debugging information");
  m.def("debug", &debug_only, "configure debugging information");
  m.def("int_add", &int_add, "A function which adds two numbers");
  m.def("load_py_module", &load_py_module, "Load mlir file and return py_module");

  py::class_<quant_brief_info>(m, "q_info", "simple tensor quant info")
      .def_readwrite("dtype", &quant_brief_info::dtype)
      .def_readwrite("shape", &quant_brief_info::shape)
      .def_readwrite("scale", &quant_brief_info::scale)
      .def_readwrite("zp", &quant_brief_info::zp);

  // clang-format off
  py::class_<py_module>(m, "py_module", "MLIR Module")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def(py::init<py::capsule>())
      .def("load", &py_module::load, "load module from IR")
      .def("set_weight_npz", &py_module::set_weightFilePath_, "set path to the weight npz")
      .def("set_tensor", &py_module::set_tensor)
      .def("set_tensor_from_int", &py_module::set_tensor_from_int)
      .def("get_tensor", &py_module::get_tensor, "get one tensor data")
      .def("get_fp32_tensor", &py_module::get_fp32_tensor, "get one fp32 tensor data")
      .def("get_all_tensor", &py_module::getAllTensor, "dump all tensor data")
      .def("invoke", &py_module::invoke)
      .def("fake_quant_weight", &py_module::fake_quant_weight)
      .def("invoke_at", &py_module::invoke_at, "invote at specified layer")
      // .def("backward_weight_at", &py_module::backward_weight_at, "invoke the backward weight function of conv op")
      .def("invoke_from", &py_module::invoke_from, "invote from specified layer to the end")
      .def("get_tensor_qinfo", &py_module::format_tensor_qinfo, "get simple quant info of tensor")
      .def_readonly("input_names", &py_module::input_names)
      .def_readonly("output_names", &py_module::output_names)
      .def_readonly("all_tensor_names", &py_module::all_tensor_names)
      .def_readonly("all_weight_names", &py_module::all_weight_names)
      .def_readonly_static("version", &py_module::version);
  // clang-format on
}
