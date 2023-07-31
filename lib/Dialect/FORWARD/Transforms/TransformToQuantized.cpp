//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "FORWARD/Dialect/FORWARD/Transforms/Passes.h"
#include "FORWARD/Support/MathUtils.h"
#include "FORWARD/Support/Module.h"
#include "FORWARD/Support/ModuleInterpreter.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "PassDetail.h"
#include <fstream>
#include <regex>
#include <sstream>

using namespace llvm;

namespace mlir {


// typedef struct {
//   double threshold;
//   double min;
//   double max;
// } cali_info;
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//


void populateFORWARDQuantizationConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
        InputLowering,
        AddLowering,
        ConvLowering,
        AvgPoolLowering,
        MaxPoolLowering,
        SoftmaxLowering
      // clang-format on
      >(patterns->getContext());
}

//===------------------------------------------------------------===//
// InputLowering
//===------------------------------------------------------------===//
void InputLowering::Lowering(PatternRewriter &rewriter, tosa::ConstOp op) const {
  assert(op->getNumResults() == 1);
  auto outType = change_dataformat(op->getResult(0).getType());
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));
  std::vector<int32_t> perms = {0, 2, 3, 1};
  auto const_ty = RankedTensorType::get({4}, rewriter.getI32Type());
  DenseElementsAttr attr = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(perms.data(), perms.size()));
  auto constop =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);
  operands.push_back(constop->getResult(0));
  rewriter.replaceOpWithNewOp<mlir::tosa::TransposeOp>(op, outType, operands);
}

//===------------------------------------------------------------===//
// AddLowering
//===------------------------------------------------------------===//
void AddLowering::Lowering(PatternRewriter &rewriter, tosa::AddOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = change_dataformat(op->getResult(0).getType());
  auto coeff = op.getCoeffAttr();
  // TODO: coeff -> constOp
  /*
  if (!coeff) {
    float coeff0 =
  coeff.getValue()[0].cast<mlir::FloatAttr>().getValueAsDouble();

    auto const_ty = RankedTensorType::get({}, rewriter.getI32Type());
    DenseElementsAttr attr = DenseElementsAttr::get(const_ty,
                      llvm::ArrayRef(perms.data(), perms.size()));
    auto constop = rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty,
  attr); double coeff1 =
  coeff.getValue()[1].cast<mlir::FloatAttr>().getValueAsDouble();
  }
  */
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    operands.push_back(in);
  }
  // do_relu
  if (op.getDoRelu()) {
    // Add op
    auto add =
        rewriter.create<mlir::tosa::AddOp>(op->getLoc(), newType, operands);
    auto relu_limit = op.getReluLimit();
    std::vector<NamedAttribute> clamp_attr =
        gen_clamp_attr(rewriter, newType, relu_limit);
    auto out_type = add->getResult(0).getType();
    // Clamp op
    auto clamp = rewriter.create<mlir::tosa::ClampOp>(
        op->getLoc(), out_type, add->getResults(), clamp_attr);
    rewriter.replaceOp(op, clamp->getResults());
  } else {
    rewriter.replaceOpWithNewOp<mlir::tosa::AddOp>(op, newType, operands);
  }
}

//===------------------------------------------------------------===//
// ConvLowering
//===------------------------------------------------------------===//
void ConvLowering::Lowering(PatternRewriter &rewriter, tosa::Conv2DOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = change_dataformat(op->getResult(0).getType());
  std::vector<NamedAttribute> attrs;
  auto pads = module::getI64Array(op.getPads());
  std::vector<int64_t> newValues{pads->at(0), pads->at(2), pads->at(1),
                                 pads->at(3)};
  attrs.push_back(
      rewriter.getNamedAttr("pad", rewriter.getDenseI64ArrayAttr(newValues)));
  auto strides = module::getI64Array(op.getStrides());
  attrs.push_back(
      rewriter.getNamedAttr("stride", rewriter.getDenseI64ArrayAttr(*strides)));
  auto dilations = module::getI64Array(op.getDilations(), 2, 1);
  attrs.push_back(rewriter.getNamedAttr(
      "dilation", rewriter.getDenseI64ArrayAttr(*dilations)));
  std::vector<Value> operands;
  auto ic = op->getOperand(0).getType().cast<RankedTensorType>().getShape()[1];
  auto oc = op->getResult(0).getType().cast<RankedTensorType>().getShape()[1];
  auto kc = op->getOperand(1).getType().cast<RankedTensorType>().getShape()[1];
  auto group = op.getGroup();
  // depth_wise conv
  if (ic == oc && oc == group && kc == 1) {
    // auto weight = op->getOperand(1).getDefiningOp()->getResult(0);
    auto weight = op->getOperand(1);
    auto weightTy = weight.getType().cast<RankedTensorType>(); // NCHW
    // NCHW -> HWCM(HWCN)  In this case, "N"->"C", "C"="M"=1
    // std::vector<int32_t> perms = {2, 3, 0, 1};
    // NHWC -> HWCM(HWCN)  In this case, "N"->"C", "C"="M"=1
    std::vector<int32_t> perms = {1, 2, 0, 3};
    auto const_ty = RankedTensorType::get({4}, rewriter.getI32Type());
    DenseElementsAttr attr = DenseElementsAttr::get(
        const_ty, llvm::ArrayRef(perms.data(), perms.size()));
    auto constop =
        rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);
    std::vector<int64_t> newWeightShape;
    auto weightShape = weightTy.getShape(); // NCHW
    newWeightShape.push_back(weightShape[2]);
    newWeightShape.push_back(weightShape[3]);
    newWeightShape.push_back(weightShape[0]);
    newWeightShape.push_back(weightShape[1]); // HWCM(HWCN)
    auto newWeightTy =
        RankedTensorType::get(newWeightShape, weightTy.getElementType());
    auto newweight =
        rewriter
            .create<mlir::tosa::TransposeOp>(op->getLoc(), newWeightTy, weight,
                                             constop->getResult(0))
            ->getResult(0);
    operands.push_back(op->getOperand(0));
    operands.push_back(newweight);
    for (unsigned i = 2; i < op->getNumOperands(); i++) {
      operands.push_back(op->getOperand(i));
    }
    // do_relu
    if (op.getDoRelu()) {
      // Conv op
      auto conv = rewriter.create<mlir::tosa::DepthwiseConv2DOp>(
          op->getLoc(), newType, operands, attrs);
      auto relu_limit = op.getReluLimit();
      std::vector<NamedAttribute> clamp_attr =
          gen_clamp_attr(rewriter, newType, relu_limit);
      auto out_type = conv->getResult(0).getType();
      // Clamp op
      auto clamp = rewriter.create<mlir::tosa::ClampOp>(
          op->getLoc(), out_type, conv->getResults(), clamp_attr);
      rewriter.replaceOp(op, clamp->getResults());
    } else {
      rewriter.replaceOpWithNewOp<mlir::tosa::DepthwiseConv2DOp>(
          op, newType, operands, attrs);
    }
  }
  // normal conv
  else if (group == 1) {
    for (auto in : op->getOperands()) {
      operands.push_back(in);
    }
    // do_Relu
    if (op.getDoRelu()) {
      // Conv op
      auto conv = rewriter.create<mlir::tosa::Conv2DOp>(op->getLoc(), newType,
                                                        operands, attrs);
      auto relu_limit = op.getReluLimit();
      std::vector<NamedAttribute> clamp_attr =
          gen_clamp_attr(rewriter, newType, relu_limit);
      auto out_type = conv->getResult(0).getType();
      // Clamp op
      auto clamp = rewriter.create<mlir::tosa::ClampOp>(
          op->getLoc(), out_type, conv->getResults(), clamp_attr);
      rewriter.replaceOp(op, clamp->getResults());
    } else {
      rewriter.replaceOpWithNewOp<mlir::tosa::Conv2DOp>(op, newType, operands,
                                                        attrs);
    }
  }
  // TODO: support for group conv
  else
    ;
}

//===------------------------------------------------------------===//
// AvgPoolLowering
//===------------------------------------------------------------===//
void AvgPoolLowering::Lowering(PatternRewriter &rewriter,
                               tosa::AvgPoolOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = change_dataformat(op->getResult(0).getType());
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    operands.push_back(in);
  }
  std::vector<NamedAttribute> attrs;
  auto strides = module::getI64Array(op.getStrides());
  attrs.push_back(
      rewriter.getNamedAttr("stride", rewriter.getDenseI64ArrayAttr(*strides)));
  auto kernels = module::getI64Array(op.getKernelShape());
  attrs.push_back(
      rewriter.getNamedAttr("kernel", rewriter.getDenseI64ArrayAttr(*kernels)));
  auto x1 =
      op.getPadsAttr().getValue()[0].cast<mlir::IntegerAttr>().getInt(); // top
  auto x2 = op.getPadsAttr()
                .getValue()[2]
                .cast<mlir::IntegerAttr>()
                .getInt(); // bottom
  auto x3 =
      op.getPadsAttr().getValue()[1].cast<mlir::IntegerAttr>().getInt(); // left
  auto x4 = op.getPadsAttr()
                .getValue()[3]
                .cast<mlir::IntegerAttr>()
                .getInt(); // right
  std::vector<int64_t> newValues{x1, x2, x3, x4};
  attrs.push_back(
      rewriter.getNamedAttr("pad", rewriter.getDenseI64ArrayAttr(newValues)));
  attrs.push_back(
      rewriter.getNamedAttr("acc_type", TypeAttr::get(rewriter.getF32Type())));
  // do_relu
  if (op.getDoRelu()) {
    // Avgpool op
    auto avgpool = rewriter.create<mlir::tosa::AvgPool2dOp>(
        op->getLoc(), newType, operands, attrs);
    auto relu_limit = op.getReluLimit();
    std::vector<NamedAttribute> clamp_attr =
        gen_clamp_attr(rewriter, newType, relu_limit);
    auto out_type = avgpool->getResult(0).getType();
    // Clamp op
    auto clamp = rewriter.create<mlir::tosa::ClampOp>(
        op->getLoc(), out_type, avgpool->getResults(), clamp_attr);
    rewriter.replaceOp(op, clamp->getResults());
  } else {
    rewriter.replaceOpWithNewOp<mlir::tosa::AvgPool2dOp>(op, newType, operands,
                                                         attrs);
  }
}

//===------------------------------------------------------------===//
// MaxPoolLowering
//===------------------------------------------------------------===//
void MaxPoolLowering::Lowering(PatternRewriter &rewriter,
                               tosa::MaxPoolOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = change_dataformat(op->getResult(0).getType());
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    operands.push_back(in);
  }
  std::vector<NamedAttribute> attrs;
  auto strides = module::getI64Array(op.getStrides());
  attrs.push_back(
      rewriter.getNamedAttr("stride", rewriter.getDenseI64ArrayAttr(*strides)));
  auto kernels = module::getI64Array(op.getKernelShape());
  attrs.push_back(
      rewriter.getNamedAttr("kernel", rewriter.getDenseI64ArrayAttr(*kernels)));
  auto x1 =
      op.getPadsAttr().getValue()[0].cast<mlir::IntegerAttr>().getInt(); // top
  auto x2 = op.getPadsAttr()
                .getValue()[2]
                .cast<mlir::IntegerAttr>()
                .getInt(); // bottom
  auto x3 =
      op.getPadsAttr().getValue()[1].cast<mlir::IntegerAttr>().getInt(); // left
  auto x4 = op.getPadsAttr()
                .getValue()[3]
                .cast<mlir::IntegerAttr>()
                .getInt(); // right
  std::vector<int64_t> newValues{x1, x2, x3, x4};
  attrs.push_back(
      rewriter.getNamedAttr("pad", rewriter.getDenseI64ArrayAttr(newValues)));
  // do_relu
  if (op.getDoRelu()) {
    // Maxpool op
    auto maxpool = rewriter.create<mlir::tosa::MaxPool2dOp>(
        op->getLoc(), newType, operands, attrs);
    auto relu_limit = op.getReluLimit();
    std::vector<NamedAttribute> clamp_attr =
        gen_clamp_attr(rewriter, newType, relu_limit);
    auto out_type = maxpool->getResult(0).getType();
    // Clamp op
    auto clamp = rewriter.create<mlir::tosa::ClampOp>(
        op->getLoc(), out_type, maxpool->getResults(), clamp_attr);
    rewriter.replaceOp(op, clamp->getResults());
  } else {
    rewriter.replaceOpWithNewOp<mlir::tosa::MaxPool2dOp>(op, newType, operands,
                                                         attrs);
  }
}

//===------------------------------------------------------------===//
// SoftmaxLowering
//===------------------------------------------------------------===//
void SoftmaxLowering::Lowering(PatternRewriter &rewriter,
                               tosa::SoftmaxOp op) const {
  assert(op->getNumResults() == 1);
  auto preType = op->getResult(0).getType();
  auto newType = change_dataformat(preType);
  auto size = preType.cast<RankedTensorType>().getShape().size();
  int32_t new_axis, axis = op.getAxis();
  if (size == 4) {
    if (axis == 1 || axis == -3)
      new_axis = 3; // C
    else if (axis == 2 || axis == -2)
      new_axis = 1; // H
    else if (axis == 3 || axis == -1)
      new_axis = 2; // W
    else
      new_axis = axis; // N
  }
  bool log_attr_val = op.getLog();
  // op.getBeta() (beta = 1 by default)
  // ReduceMaxOp
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(new_axis)));
  std::vector<int64_t> out_shape(newType.cast<RankedTensorType>().getShape());
  out_shape[new_axis] = 1;
  auto out_type = RankedTensorType::get(
      out_shape, newType.cast<RankedTensorType>().getElementType());
  auto reducemax = rewriter.create<mlir::tosa::ReduceMaxOp>(
      op->getLoc(), out_type, op->getOperands(), attrs);
  // SubOp
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));
  operands.push_back(reducemax->getResult(0));
  auto sub =
      rewriter.create<mlir::tosa::SubOp>(op->getLoc(), newType, operands);
  // ExpOp
  auto sub_ty = sub->getResult(0).getType();
  auto exp = rewriter.create<mlir::tosa::ExpOp>(op->getLoc(), sub_ty,
                                                sub->getResults());
  // ReduceSumOp ( out_type & attrs same as ReduceMaxOp)
  auto reducesum = rewriter.create<mlir::tosa::ReduceSumOp>(
      op->getLoc(), out_type, exp->getResults(), attrs);
  // LogSoftmax ? Softmax ?
  if (log_attr_val) {
    // LogOp
    auto reducesum_ty = reducesum->getResult(0).getType();
    auto log = rewriter.create<mlir::tosa::LogOp>(op->getLoc(), reducesum_ty,
                                                  reducesum->getResults());
    // SubOp
    operands.clear();
    operands.push_back(sub->getResult(0));
    operands.push_back(log->getResult(0));
    auto sub2 =
        rewriter.create<mlir::tosa::SubOp>(op->getLoc(), newType, operands);
    rewriter.replaceOp(op, sub->getResults());
  } else {
    // ReciprocalOp
    auto reducesum_ty = reducesum->getResult(0).getType();
    auto reciprocal = rewriter.create<mlir::tosa::ReciprocalOp>(
        op->getLoc(), reducesum_ty, reducesum->getResults());
    // MulOp
    auto mul = rewriter.create<mlir::tosa::MulOp>(
        op->getLoc(), newType, exp->getResult(0), reciprocal->getResult(0),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, mul->getResults());
  }
}


namespace FORWARD {

class TransformToQuantizedPass
    : public TransformToQuantizedBase<TransformToQuantizedPass> {
public:
  TransformToQuantizedPass() {}
  void runOnOperation() override {
  module_ = getOperation();
  ctx_ = &getContext();
  mainFunc_ = module::getMainFuncOp();

  RewritePatternSet patterns(ctx_);
  ConversionTarget target(*ctx_);
  target.addLegalDialect<mlir::tosa::TosaDialect, mlir::func::FuncDialect>();

    // patterns.add<LowerTopWeightOp>(patterns.getContext(), includeWeight);
    populateTopToTosaConversionPatterns(&patterns);
    auto config = GreedyRewriteConfig();
    config.maxIterations = 1;
    applyPatternsAndFoldGreedily(module_, std::move(patterns), config);

  }
protected:
  ModuleOp module_;
  FuncOp mainFunc_;
  MLIRContext *ctx_;
};

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createTransformToQuantizedPass() {
  return std::make_unique<TransformToQuantizedPass>();
}

} // namespace FORWARD
}
