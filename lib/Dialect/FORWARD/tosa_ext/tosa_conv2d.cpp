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

conv_attr_t conv2d_parseParam(Operation* op) {
  conv_attr_t p = {0};
  auto tosa_Conv2D = dyn_cast_or_null<tosa::Conv2DOp>(op);
  assert(tosa_Conv2D != nullptr);
  
  p.id = p.od = p.kd = p.sd = p.dd = 1;
  // auto i_s = getInput().getType().cast<RankedTensorType>().getShape();
  auto i_s = op->getOperand(0).getType().cast<RankedTensorType>().getShape();
  // auto o_s = getOutput().getType().cast<RankedTensorType>().getShape();
  auto o_s = op->getResults()[0].getType().cast<RankedTensorType>().getShape();
  p.do_relu = false;
  // p.do_relu = getDoRelu();
  // p.relu_limit = getReluLimit().convertToDouble();
  // p.has_bias = getWithBias();
  p.has_bias = true;
  p.dims = i_s.size() - 2;
  p.n = i_s[0];
  p.ic = i_s[3];
  p.ih = i_s.size() > 2 ? i_s[1] : 1;
  p.iw = i_s.size() > 3 ? i_s[2] : 1;
  p.oc = o_s[3];
  p.oh = o_s.size() > 2 ? o_s[1] : 1;
  p.ow = o_s.size() > 3 ? o_s[2] : 1;
  
  // auto kernel = module::getI64Array(getKernelShape());
  auto kernel = tosa_Conv2D.getWeight().getType().cast<RankedTensorType>().getShape();
  p.kh = kernel[1];
  p.kw = kernel[2];
  
  // auto pads_v = module::getI64Array(getPads());
  auto pads_v = tosa_Conv2D.getPad();
  p.pht = pads_v[0];
  p.pwl = pads_v[1];
  p.phb = pads_v[2];
  p.pwr = pads_v[3];

  if (module::isUniformQuantized(tosa_Conv2D.getInput())) {
    p.pad_value = module::getUniformQuantizedType(tosa_Conv2D.getInput()).getZeroPoint();
  }
  // p.kernel_zp = getKernelZp();

  // auto strides_v = module::getI64Array(getStrides());
  auto strides_v = tosa_Conv2D.getStride();
  p.sh = strides_v[0];
  p.sw = strides_v[1];
  // auto dhdw = module::getI64Array(getDilations(), 2, 1);
  auto dhdw = tosa_Conv2D.getDilation();
  p.dh = dhdw[0];
  p.dw = dhdw[1];
  // auto ins = module::getI64Array(getInserts(), 2, 0);
  // p.ins_h = ins->at(0);
  // p.ins_w = ins->at(1);
  p.ins_h = 0;
  p.ins_w = 0;
  // p.groups = getGroup();
  p.groups = 1;
  // p.is_dw = (p.oc == p.ic && p.oc == p.groups && p.groups > 1);
  p.is_dw = 0;
  return p;
}

void tosa_conv2d_infer(Operation* op, mlir::FORWARD::ModuleInterpreter* interpreter) {
  /// tensor
  llvm::errs() <<"\ntosa_conv2d_infer!!! \n"; 

  std::shared_ptr<std::vector<float>> tensor = interpreter->mem_map[module::getName(op).str()];

  auto p = *(interpreter->inference_map[module::getName(op).str()]);

  auto conv = new Conv();
  auto attr = conv2d_parseParam(op);
  conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], attr);
  p.handle = (void *)conv;

  auto tosa_Conv2D = dyn_cast_or_null<tosa::Conv2DOp>(op);
  
  auto num_elem = module::getNumElements(tosa_Conv2D.getOutput());

  // float* output_cpy;
  // mempcpy(output_cpy, p.outputs[0], num_elem);

  assert(p.handle != nullptr);
  auto conv_handle = (Conv *)p.handle;
  conv_handle->run();

  // for(int i = 0; i < num_elem; i++){
  //   llvm::errs() << "line" << i << ": " << output_cpy[i] << " " << p.outputs[0][i] << "\n";
  // }

  auto out_type = module::getStorageType(tosa_Conv2D.getOutput());
  // auto num_elem = module::getNumElements(tosa_Conv2D.getOutput());
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } 
  /*
  else if (module::isUniformQuantized(tosa_Conv2D.getOutput())) {
    int64_t n, c, h, w;
    auto sType = module::getStorageType(tosa_Conv2D.getOutput());
    module::getNCHW(getOutput(), n, c, h, w);
    auto o_qtype = module::getUniformQuantizedType(getOutput());
    auto rshift_v = module::getI64Array(getRshift().value());
    auto multiplier_v =
        module::getI64Array(getMultiplier(), rshift_v->size(), 1);
    bool per_axis = rshift_v->size() == c;
    // do bias after conv prevent precision issue
    auto bias_i32 = std::make_shared<std::vector<int32_t>>(c, 0);
    // bool do_relu = getDoRelu();
    bool do_relu = false;
    if (getWithBias()) {
      auto biasOp = cast<tosa::ConstOp>(getBias().getDefiningOp());
      bias_i32 = biasOp.read_as_int32();
    }
    auto qmode = getQuantMode();
    bool is_tf = qmode == tpu::RequantMode::QDM ||
                 qmode == tpu::RequantMode::TFLite ||
                 qmode == tpu::RequantMode::TFLite_LShift;
    auto rmode = is_tf ? ROUNDING_HALF_AWAY_FROM_ZERO : ROUNDING_HALF_UP;

#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int ic = 0; ic < c; ic++) {
      int64_t shift = per_axis ? rshift_v->at(ic) : rshift_v->at(0);
      int64_t multi = 1;
      if (qmode != tpu::RequantMode::OnlyShift) {
        multi = per_axis ? multiplier_v->at(ic) : multiplier_v->at(0);
      }
      int32_t bias = bias_i32->at(ic);
      for (int in = 0; in < n; in++) {
        for (int hw = 0; hw < h * w; hw++) {
          int offset = (in * c + ic) * h * w + hw;
          int64_t v = 0;
          int64_t tmp = p.outputs[0][offset] + bias;
          v = applyMultiplierAndRShift(tmp, multi, shift, qmode, rmode) +
              o_qtype.getZeroPoint();
          if (do_relu && (v < 0)) {
            v = 0;
          }
          p.outputs[0][offset] = saturate(v, out_type);
        }
      }
    }
  }
  */
}


}
}