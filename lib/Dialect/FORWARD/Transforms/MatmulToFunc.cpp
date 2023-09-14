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
#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "PassDetail.h"
#include <fstream>
#include <cfloat>
#include <regex>
#include <sstream>

using namespace llvm;


namespace mlir {
namespace FORWARD {

struct Matrix_Shape{
  int dim0, dim1, dim2;
};

SmallVector<tosa::MatMulOp, 8> toFuncList;

int FindCutFactor(tosa::MatMulOp MMop){
  mlir::Operation* inputA = MMop.getA().getDefiningOp();
  mlir::Operation* inputB = MMop.getB().getDefiningOp();
  RankedTensorType Atype = MMop.getA().getType().cast<RankedTensorType>();
  RankedTensorType Btype = MMop.getB().getType().cast<RankedTensorType>();
  RankedTensorType Outtype = MMop.getResult().getType().cast<RankedTensorType>();
  assert(Atype && Btype && Outtype);
  llvm::ArrayRef<int64_t> Ashape = Atype.getShape();
  llvm::ArrayRef<int64_t> Bshape = Btype.getShape();
  llvm::ArrayRef<int64_t> Outshape = Outtype.getShape();

  // Check matrix's shape
  if(Ashape.size() == 2){
    assert(0 && "Todo: no support 2 dim.");
  }
  if(Ashape.size() != 3 && Ashape.size() != 2){
    return 0;
  }
  if(Ashape[0] != 1 || Bshape[0] != 1 || Outshape[0] != 1){
    return 0;
  }

  if(Ashape.size() == 3){
    assert(Ashape[2] == Bshape[1]);
  }

  int64_t cut_factor, toCut = Ashape[2];

  // cut could be 32,16,8,4,2,1
  if(toCut % 32 == 0){
    cut_factor = 32;
  }
  else if(toCut % 16 == 0){
    cut_factor = 16;
  }
  else if(toCut % 8 == 0){
    cut_factor = 8;
  }
  else if(toCut % 4 == 0){
    cut_factor = 4;
  }
  else{
    cut_factor = 0;
  }

  if(Outshape[2] % cut_factor != 0){
    return 0;
  }
  else{
    return cut_factor;
  }

  // llvm::errs() << "Ashape: " ;
  // for(int64_t dim : Ashape){
  //   llvm::errs()  << dim <<" ";
  // }
  // llvm::errs() << "\n" ;

  // llvm::errs() << "Bshape: " ;
  // for(int64_t dim : Bshape){
  //   llvm::errs()  << dim <<" ";
  // }
  // llvm::errs() << "\n" ;

  // llvm::errs() << "Outshape: " ;
  // for(int64_t dim : Outshape){
  //   llvm::errs()  << dim <<" ";
  // }
  // llvm::errs() << "\n" ;

}

SmallVector<int, 3> cutShape(llvm::ArrayRef<int64_t> shape, int cut){
  assert(shape.size() == 3);
  assert(shape[0] == 1);
  assert(shape[2] % cut == 0);

  SmallVector<int, 3> new_shape;
  new_shape.push_back(shape[2] / cut);
  new_shape.push_back(shape[1]);
  new_shape.push_back(cut);
  
  return new_shape;
}

std::pair<int, int> adjustScale(double scale, int shift_min, int shift_max) {
	double diff = DBL_MAX;
	int n, m;
	for (int i = 0; i < 256; i++) {
		for (int j = shift_min; j <= shift_max; j++) {
			auto temp = abs(scale - i * pow(2, -j));
			if (temp <= diff) {
				n = i;
				m = j;
				diff = temp;
			}
		}
	}
	return std::make_pair(n, m);
}

func::FuncOp GenMatmulFunc(OpBuilder builder, tosa::MatMulOp MMop, int MMid)
{
  Location loc = MMop.getLoc();
  // Create a builder with no insertion point, insertion will happen separately
  // due to symbol table manipulation
  // // Contains the region of code that will be outlined
  // Region &KernelOpBody = KernelOp.body();
  std::string FnName = "Matmul_accel_" + std::to_string(MMid);
  // errs() << kernelFnName << ":\n";
  // KernelOp.dump();
  // Identify uses from values defined outside of the scope of the launch
  // operation.
  // getUsedValuesDefinedAbove(KernelOpBody, operands);

  // Create the func.func operation.
  SmallVector<Type, 4> FnOperandTypes;
  FnOperandTypes.push_back(MMop.getA().getType());
  FnOperandTypes.push_back(MMop.getB().getType());
  FnOperandTypes.push_back(MMop.getResult().getType());
  // kernelOperandTypes.reserve(operands.size());
  // for (Value operand : operands)
  // {
  //   // errs()  << "  operands:"; operand.dump();
  //   kernelOperandTypes.push_back(operand.getType());
  // }
  FunctionType type =
      FunctionType::get(MMop.getContext(), FnOperandTypes, {});
  func::FuncOp MatmulFunc = builder.create<func::FuncOp>(loc, FnName, type);
  //  std::cout << "[debug] after create:\n"; KernelFunc.dump();
  MatmulFunc->setAttr(FnName, builder.getUnitAttr());

  return MatmulFunc;
}

class MatmulToFuncPass
    : public MatmulToFuncBase<MatmulToFuncPass> {
public:
  MatmulToFuncPass() {}
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m);
    int matmul_idx = 0;
    m.walk([&](tosa::MatMulOp MMop){
      RankedTensorType Atype = MMop.getA().getType().cast<RankedTensorType>();
      RankedTensorType Btype = MMop.getA().getType().cast<RankedTensorType>();
      RankedTensorType Outtype = MMop.getResult().getType().cast<RankedTensorType>();
      assert(Atype && Btype && Outtype);

      int cut = FindCutFactor(MMop);

      if(cut == 0){
        /// not suitable to use the hardware accelerator 
        return WalkResult::advance();
      }
      toFuncList.push_back(MMop);
      return WalkResult::advance();
    });

    for(tosa::MatMulOp& MMop : toFuncList){
      // llvm::errs() << "Mmop: "<< MMop << "\n";
      // This matmul is suitable to use the hardware accelerator.
      // change it to function call
      auto tensorEmpty = builder.create<tensor::EmptyOp>(
          MMop.getLoc(), 
          MMop.getResult().getType().cast<RankedTensorType>().getShape(),
          MMop.getResult().getType().cast<RankedTensorType>().getElementType());
       MMop.getOperation()->replaceAllUsesWith(tensorEmpty);

      SmallVector<Value, 3> operands;
      operands.push_back(MMop.getA());
      operands.push_back(MMop.getB());
      operands.push_back(tensorEmpty.getResult());

      func::FuncOp MatmulFunc = GenMatmulFunc(builder, MMop, matmul_idx++);
      MatmulFunc.setVisibility(mlir::SymbolTable::Visibility::Private);
      m.getBodyRegion().front().push_back(MatmulFunc);
      // llvm::errs() << "MatmulFunc: "<< MatmulFunc << "\n";
      func::CallOp call = builder.create<func::CallOp>(MMop.getLoc(), MatmulFunc, operands);
      MMop.getOperation()->getBlock()->push_back(tensorEmpty);
      MMop.getOperation()->getBlock()->push_back(call);
      tensorEmpty.getOperation()->moveAfter(MMop);
      call.getOperation()->moveAfter(tensorEmpty);

      // llvm::errs() << "call: "<< call << "\n";
      // llvm::errs() << "m: "<< m << "\n";
      /// cut shapes
      mlir::Operation* inputA = MMop.getA().getDefiningOp();
      mlir::Operation* inputB = MMop.getB().getDefiningOp();
      RankedTensorType Atype = MMop.getA().getType().cast<RankedTensorType>();
      RankedTensorType Btype = MMop.getB().getType().cast<RankedTensorType>();
      RankedTensorType Outtype = MMop.getResult().getType().cast<RankedTensorType>();
      assert(Atype && Btype && Outtype);
      llvm::ArrayRef<int64_t> Ashape = Atype.getShape();
      llvm::ArrayRef<int64_t> Bshape = Btype.getShape();
      llvm::ArrayRef<int64_t> Outshape = Outtype.getShape();
      SmallVector<int, 3> cutAshape, cutBshape, cutOutshape;
      int cut = FindCutFactor(MMop);
      cutAshape = cutShape(Ashape, cut);
      // exchange Bshape 's dim 1 and 2
      std::vector<int64_t> mutableBshape(Bshape.begin(), Bshape.end());
      assert(mutableBshape.size() == 3);
      int64_t tmp_shape2 = mutableBshape[2];
      mutableBshape[2] = mutableBshape[1];
      mutableBshape[1] = tmp_shape2;

      cutBshape = cutShape(mutableBshape, cut);
      cutOutshape = cutShape(Outshape, cut);
    // llvm::errs() << "cutBshape0:" << cutBshape[0]
    //            << ",1:" << cutBshape[1]
    //            << ",2:"  << cutBshape[2]<< "\n" ;
      /// Calculate the m and exp
      double threshold = 13.25392; /// Our customized defination
      double scale = 127 / 13.25392;
      std::pair<int, int> m_exp;
      m_exp = adjustScale(scale, /*qbits*/ 8, /* 2*qbits+5 */ 2*8+5);

      /// add input0&input1&output shape and m_exp to Matmul function
      DenseIntElementsAttr shape_attr;
      shape_attr = builder.getI32VectorAttr(cutAshape);
      MatmulFunc->setAttr("input0", shape_attr);
      shape_attr = builder.getI32VectorAttr(cutBshape);
      MatmulFunc->setAttr("input1", shape_attr);
      shape_attr = builder.getI32VectorAttr(cutOutshape);
      MatmulFunc->setAttr("output", shape_attr);
      IntegerAttr m = builder.getI64IntegerAttr(m_exp.first);
      MatmulFunc->setAttr("m", m);
      IntegerAttr exp = builder.getI64IntegerAttr(m_exp.second);
      MatmulFunc->setAttr("exp", exp);

      /// TODO: scale

      /// erase MatmulOp
      MMop.getOperation()->erase();
    }
  }
   
};

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createMatmulToFunc() {
  return std::make_unique<MatmulToFuncPass>();
}

} // namespace FORWARD
}
