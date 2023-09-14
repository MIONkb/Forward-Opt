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

matmul_attr_t matmul_parseParam(Operation* op) {
  matmul_attr_t p = {0};
  auto tosa_MatMul = dyn_cast_or_null<tosa::MatMulOp>(op);
  assert(tosa_MatMul != nullptr);

  auto a_s = SmallVector<int64_t>(module::getShape(tosa_MatMul.getA()));
  auto b_s = SmallVector<int64_t>(module::getShape(tosa_MatMul.getB()));
  auto o_s = SmallVector<int64_t>(module::getShape(tosa_MatMul.getC()));

  // p.input_zp = getInputZp();
  p.input_zp = 0;
  p.with_bias = false;
  p.do_relu = false;
  // p.relu_limit = this->getReluLimit().convertToDouble();
  // p.right_zp = getRightZp();
  p.right_zp = 0;
  p.hdim_is_batch = false;
  // p.left_reuse = getLeftReuse();

  int a_dims = a_s.size();
  int b_dims = b_s.size();
  int o_dims = o_s.size();
  p.right_transpose = false;
  p.left_transpose = false;
  p.output_transpose = false;
  if (b_dims == 1) {
    assert(p.right_transpose == false);
    b_s.push_back(1);
    o_s.push_back(1);
    b_dims += 1;
    o_dims += 1;
  }
  if (a_dims == 1){
    assert(p.left_transpose == false);
    a_s.insert(a_s.begin(), 1);
    o_s.insert(o_s.begin(), 1);
    a_dims += 1;
    o_dims += 1;
  }
  p.N = p.right_transpose ? b_s[b_dims - 2] : b_s[b_dims - 1];
  assert(p.N == o_s[o_dims - 1]);
  p.K = p.right_transpose ? b_s[b_dims - 1] : b_s[b_dims - 2];
  p.batch = 1;
  for (int i = 0; i < b_dims - 2; i++) {
    p.batch *= b_s[i];
  }
  if (p.batch > 1 || o_dims <= 2) {
    p.M = o_s[o_dims - 2];
  } else {
    p.M = std::accumulate(o_s.begin(), o_s.begin() + o_dims - 1, 1,
                          std::multiplies<int64_t>());
  }
  return p;
}

void tosa_matmul_infer(Operation* op, mlir::FORWARD::ModuleInterpreter* interpreter) {
  /// tensor
  llvm::errs() <<"\ntosa_matmul_infer!!! \n"; 

  std::shared_ptr<std::vector<float>> tensor = interpreter->mem_map[module::getName(op).str()];

  auto p = *(interpreter->inference_map[module::getName(op).str()]);

  auto matmul = new MatMul();
  auto attr = matmul_parseParam(op);
  // matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], attr.batch,
  //               attr.batch_low, attr.M, attr.K, attr.N, attr.do_relu, attr.relu_limit, attr.right_zp,
  //               attr.input_zp, attr.right_transpose, attr.left_transpose,
  //               attr.output_transpose, attr.hdim_is_batch);
  auto tosa_MatMul = dyn_cast_or_null<tosa::MatMulOp>(op);
  int leftNum = module::getNumElements(tosa_MatMul.getOperand(0));
  int rightNum = module::getNumElements(tosa_MatMul.getOperand(1));


  // Open the text file
  std::ifstream inputFile0("/home/xcgao/projects/forward-opt-cpp/build/firstarg.txt");
  if (!inputFile0){
    std::cerr << "Error opening file." << std::endl;
    return;
  }
  //Read the data from the file and store it in the 2D array
  std::string line;
  int index = 0;
  float num;
  float arg0[151296];
  while (std::getline(inputFile0, line)) {
    std::istringstream iss(line);
    while(iss >> num) {
      arg0[index++] = num;
    }
  }
  llvm::errs() << "index: " << index << "\n";
  //close file
  inputFile0.close();

  index = 0;
  // Open the text file
  std::ifstream inputFile1("/home/xcgao/projects/forward-opt-cpp/build/secondarg.txt");
  if (!inputFile1){
    std::cerr << "Error opening file." << std::endl;
    return;
  }
  //Read the data from the file and store it in the 2D array
  float arg1[151296];
  while (std::getline(inputFile1, line)) {
    std::istringstream iss(line);
    while(iss >> num) {
      arg1[index++] = num;
      // llvm::errs() << index << "\n";
    }
  }
  llvm::errs() << "index: " << index << "\n";
  //close file
  inputFile1.close();

  
  // memcpy(p.inputs[0], arg0, 151296*sizeof(float));
  // memcpy(p.inputs[1], arg1, 151296*sizeof(float));
  
  // llvm::errs() << "left: \n";
  // for(int i = 0; i < 151296; i++){
  //   llvm::errs() << "line" << i << ": " << p.inputs[0][i] << "\n";
  // }
  // llvm::errs() << "right: \n";
  // for(int i = 0; i < 151296; i++){
  //   llvm::errs() << "line" << i << ": " << p.inputs[1][i] << "\n";
  // }

  // attr.K = 197; attr.N = 768; attr.M = 768;

  // matmul->setup(p.inputs[0], p.inputs[1], nullptr, p.outputs[0], attr.batch, 1,
  //               attr.M, attr.K, attr.N, attr.do_relu, attr.relu_limit, 0, 0, attr.right_transpose,
  //               0, 0, 0);
  float activation[589824];
  matmul->setup(arg0, arg1, nullptr, activation, attr.batch, 1,
                attr.M, attr.K, attr.N, attr.do_relu, attr.relu_limit, 0, 0, attr.right_transpose,
                0, 0, 0);             

  p.handle = (void *)matmul;
  assert(p.handle != nullptr);
  auto matmul_handle = (MatMul *)p.handle;
  matmul_handle->run();

}


}
}