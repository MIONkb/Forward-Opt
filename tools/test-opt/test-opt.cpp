//===- soda-opt.cpp ---------------------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

// #include "mlir/IR/Dialect.h"
// #include "mlir/IR/MLIRContext.h"
// #include "mlir/InitAllDialects.h"
// #include "mlir/InitAllPasses.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Pass/PassManager.h"
// #include "mlir/Support/FileUtilities.h"
// #include "mlir/Tools/mlir-opt/MlirOptMain.h"

// #include "mlir-c/Debug.h"

// #include "llvm/Support/CommandLine.h"
// #include "llvm/Support/InitLLVM.h"
// #include "llvm/Support/SourceMgr.h"
// #include "llvm/Support/ToolOutputFile.h"

// #include "FORWARD/Dialect/FORWARD/IR/FORWARD.h"
// #include "FORWARD/Dialect/FORWARD/Transforms/Passes.h"
// #include "FORWARD/Misc/Passes.h"

// #include "mlir/Dialect/Arith/Transforms/Passes.h"
// #include "mlir/Dialect/Func/Transforms/Passes.h"
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

#include "FORWARD/Support/Dnnl/Dnnl.h"

#include "cnpy.h"
#include "FORWARD/Support/CaliMath/CaliMath.h"
#include <iostream>
#include <fstream>
#include <iomanip>
// #include <numpy/arrayobject.h>


// Defined in the test directory, no public header.
// namespace mlir {
// void registerTestLoopPermutationPass();
// namespace test {

// int registerTestLinalgCodegenStrategy();
// } // namespace test
// } // namespace mlir

// // Register important linalg passes
// inline void registerLinalgPassesForSoda() {

//   mlir::registerLinalgPasses();
// }

// // Register important affine passes
// inline void registerAffinePassesForFORWARD() {

//   mlir::affine::registerAffineDataCopyGenerationPass();
//   mlir::affine::registerAffineLoopInvariantCodeMotionPass();
//   mlir::affine::registerAffineLoopTilingPass();
//   mlir::affine::registerAffineLoopFusionPass();
//   mlir::affine::registerAffineLoopUnrollPass();
//   mlir::affine::registerAffineScalarReplacementPass();
//   mlir::affine::registerAffineLoopUnrollAndJamPass();
//   mlir::affine::registerAffineLoopNormalizePass();
//   mlir::affine::registerSimplifyAffineStructuresPass();

//   // Test passes
//   mlir::registerTestLoopPermutationPass();
// }

using namespace mlir;

class ActInference{
  public:
    mlir::FORWARD::ModuleInterpreter* _module;
    std::map<std::string, std::pair<std::shared_ptr<std::vector<float>>, int>> _ref_activations; //first:data; second:use count
    std::map<std::string, std::tuple<float, float, float>> _activations_statistics; //first:min; second:max; third:abs
  public:
    ActInference(mlir::FORWARD::ModuleInterpreter* moduleInter);

    std::map<std::string, float> activation_collect_and_calc_th();

    void gen_ref_tensor(std::string opName);

    std::shared_ptr<std::vector<float>> get_ref_tensor(std::string opName);

    std::pair<std::vector<int>, float> histogram(std::shared_ptr<std::vector<float>> activation,
                                      float abs_value, int histogram_bin_num = 2048);

    std::map<std::string, float> find_threshold(std::map<std::string, std::vector<int>> histdataMap, 
                                      std::map<std::string, float> histwidthMap, int histogram_bin_num = 2048);

    void load_net_input();

  public:
    std::vector<std::string> getAllOpNames(){return _module->all_tensor_names;}
};

ActInference::ActInference(mlir::FORWARD::ModuleInterpreter* moduleInter){
  _module = moduleInter;
  _ref_activations.clear();
}

std::pair<std::vector<int>, float> ActInference::histogram(std::shared_ptr<std::vector<float>> activation,
                                      float abs_value, int histogram_bin_num){
  std::vector<float> absVec;
  std::transform(activation->begin(), activation->end(), std::back_inserter(absVec), [](int num) {
        return std::abs(num);
  });
  absVec.erase(std::remove(absVec.begin(), absVec.end(), 0), absVec.end());
  float width = abs_value / (histogram_bin_num - 1);
  std::vector<int> hist(histogram_bin_num, 0);
  if(absVec.size() > 0){
    for (const float& value : absVec) {
        int bin = static_cast<int>(std::floor(value / width + 0.5));
        if (bin >= 0 && bin < histogram_bin_num) {
            hist[bin]++;
        }
    }
  }
  // if t.size > 0:
  //     hist, _ = np.histogram(np.floor(t / width + 0.5),
  //                             bins=bin_num,
  //                             range=(0, bin_num - 1),
  //                             density=False)
  // else:
  //     hist = np.zeros(bin_num)
  // hist = hist.astype(np.int32)
  // return hist, width
  return std::make_pair(hist,width);
}

std::map<std::string, float> ActInference::find_threshold(std::map<std::string, std::vector<int>> histdataMap, 
                std::map<std::string, float> histwidthMap, int histbinNum){
  std::map<std::string, float> thresholds;
  for(auto &iter : histdataMap){
    std::string opname = iter.first;
    std::vector<int> hist = iter.second;
    llvm::errs() << "calculate op: " << opname << "\n";
    float width = histwidthMap[opname];
    int bin_num = histbinNum;
    thresholds[opname] = CaliMath::kl_diversity_hist(&(hist[0]), width, bin_num);
  }
  return thresholds;
}

std::map<std::string, float> ActInference::activation_collect_and_calc_th(){
  auto module = _module;
  std::map<std::string, std::vector<int>> histogram_data_map;
  std::map<std::string, float> histogram_width_map;
  std::map<std::string, float> thresholds_map_absmax;
  llvm::dbgs() << "[INFO] activation_collect_and_calc_th begin!!!!!" <<"\n";
  std::vector<std::string> tensor_names = module->all_tensor_names;

  // for(int i = 0; i < input_num; i++) //for cali images
  for(auto name:tensor_names){
    gen_ref_tensor(name);
  
    //define min/max
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    float abs_value = 0;

    auto activation = get_ref_tensor(name);
    auto act_max = *(std::max_element(activation->begin(), activation->end()));
    auto act_min = *(std::min_element(activation->begin(), activation->end()));
    min_value = std::min(act_min, min_value);
    max_value = std::max(act_max, max_value);
    abs_value = std::max(abs(min_value), abs(max_value));
    if(abs_value <= 1e-5){
      // if op's outputs are all close to zero, change it to 1e-5 for them.
      min_value = -1e-5;
      max_value = 1e-5;
      abs_value = 1e-5;
      llvm::dbgs() << "[WARNING] layer " << "opname" << " is all zeros. Please check the input data correctness.\n";
    }
    _activations_statistics[name] = std::make_tuple(min_value, max_value, abs_value);
    // abs_value = std::get<2>(_activations_statistics[name]);
    auto histogramParam = histogram(activation, abs_value);
    auto hist = histogramParam.first;
    float width = histogramParam.second;
    if(histogram_data_map.find(name) == histogram_data_map.end()){
      histogram_data_map[name] = hist;
      histogram_width_map[name] = width;
    }else{
      // histogram_data_map[name] += hist;
      histogram_data_map[name].insert(histogram_data_map[name].end(), hist.begin(), hist.end());
    }
  }

  auto thresholds_map = find_threshold(histogram_data_map, histogram_width_map);
  for(auto &iter : _activations_statistics){
    std::string opname = iter.first;
    float abs_val = std::get<2>(iter.second);
    thresholds_map_absmax[opname] = abs_val;
    if(thresholds_map[opname] > abs_val){
      thresholds_map[opname] = abs_val;
    }
  }
  return thresholds_map;

}


void ActInference::gen_ref_tensor(std::string opName){
  if(_ref_activations.find(opName) != _ref_activations.end()){
    return;
  }
  llvm::errs() << "[INFO] generate reference tensor of " << opName << "\n";
  Value curr_opValue = _module->value_map[opName];
  curr_opValue.dump();
  mlir::Operation *curr_op = curr_opValue.getDefiningOp();
  auto prevOps = curr_op->getOperands();
  auto weight_names = _module->all_weight_names;
  for(auto prevOp:prevOps){
    // prevOp.dump();
    // std::string prevOpName = _module->name_map[&prevOp];
    std::string prevOpName;
    for(auto iter: _module->value_map){
      if(iter.second == prevOp){
        prevOpName = iter.first;
        break;
      }
    }
    if(std::find(weight_names.begin(), weight_names.end(), prevOpName) != weight_names.end()){
      llvm::dbgs() << "[INFO] current previrous op " << prevOpName << " is constant\n";
      continue;
    }
    llvm::dbgs() << "[INFO] current previrous op name: " << prevOpName <<"\n";
    // std::shared_ptr<std::vector<float>> data = std::make_shared<std::vector<float>>();
    auto data = _ref_activations[prevOpName].first;
    // llvm::dbgs() << "before setTensor\n";
    // llvm::dbgs() << "data size is: " << data->size() <<"\n";
    _module->setTensor(prevOpName, data, data->size()* sizeof(float));
  }

  if(prevOps.size() > 0){
    std::shared_ptr<std::vector<float>> value = _module->invoke_at(opName);
    _ref_activations[opName] = std::make_pair(value, 7);
  }
}

std::shared_ptr<std::vector<float>> ActInference::get_ref_tensor(std::string opName){
  if(_ref_activations.find(opName) != _ref_activations.end()){
    return _ref_activations[opName].first;
  }else{
    llvm::errs() << "error " << opName << " not in ref_activations\n";
    return nullptr;
  }
}

void ActInference::load_net_input(){
  llvm::dbgs() << "[INFO] load_net_input begin\n";
  // assert(_module->input_names.size() != 0);
  for(std::string input : _module->input_names){
    auto mem_map = _module->mem_map;
    auto it = mem_map.find(input);
    if (it == mem_map.end()) {
      llvm::errs() << "Can't find op name: " << input << "\n";
      llvm_unreachable("Error, setTensor failed");
    }

    auto act = it->second;
    size_t size = act->size();
    std::shared_ptr<std::vector<float>> data = std::make_shared<std::vector<float>>();
    for(int i = 0; i < size; i++){
      data->push_back(i);
    }
    llvm::dbgs() << "data size is: " << data->size() * sizeof(float) <<"\n";
    _ref_activations[input] = std::make_pair(data, 7);
  }
  llvm::dbgs() << "[INFO] load_net_input end\n";
}

int main(int argc, char **argv) {
  // mlir::registerAllDialects();
  // mlir::registerAllPasses();
  using namespace mlir;
  std::unique_ptr<mlir::MLIRContext> context_;
  context_.reset();
  OwningOpRef<ModuleOp> module_OOR;
  std::string filename = "/home/xcgao/projects/forward-opt/models/visionLinear/tosa_elided.mlir";
  mlirEnableGlobalDebug(true);  
  llvm::DebugFlag = true;
    
  DialectRegistry registry;

  registry.insert<func::FuncDialect, FORWARD::FORWARDDialect,
                    quant::QuantizationDialect, memref::MemRefDialect,
                    tensor::TensorDialect, tosa::TosaDialect>();
  context_ = std::make_unique<MLIRContext>(registry);

  module_OOR = parseSourceFile<mlir::ModuleOp>(filename, context_.get());
  mlir::ModuleOp module_ = module_OOR.get();
  std::cout << "file:" << filename<< ", module: "<< module_OOR.get() <<"\n";
  module_.dump();

  mlir::FORWARD::ModuleInterpreter* interpreter_ = new mlir::FORWARD::ModuleInterpreter(module_);

  interpreter_->printValuemap();

  std::string weightFilePath_ = "/home/xcgao/projects/forward-opt/models/visionLinear/vit_origin_weight.npz";
  if(weightFilePath_!=""){
    interpreter_->weightnpz = weightFilePath_;
    // cnpy::npz_t npzFile = cnpy::npz_load(weightFilePath_);
    // for (const auto& pair : npzFile) {
    //   const std::string& arrayName = pair.first;
    //   const cnpy::NpyArray& array = pair.second;

    //   // 输出数组名
    //   std::cout << "Array name: " << arrayName << std::endl;

    //   // 输出数组形状
    //   std::cout << "Array shape: ";
    //   for (size_t dim : array.shape) {
    //       std::cout << dim << " ";
    //   }
    //   std::cout << std::endl;

    //   // 输出数组数据
    //   const float* data = array.data<float>();
    //   for (size_t i = 0; i < array.num_vals; ++i) {
    //       std::cout << data[i] << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // interpreter_->setweightnpz(path);
  }
  interpreter_->allocate_resources();
  
  interpreter_->printValuemap();
  // interpreter_->printMemmap();
  
  ActInference* infer = new ActInference(interpreter_);

  infer->load_net_input();

  
  // llvm::errs() << "mem_map[input_0]: ";
  // for(int i = 0; i < 768; i++){
  //   llvm::errs() << infer->_module->mem_map["input_0"]->at(i) << " ";
  // }
  
  auto thresholds_map = infer->activation_collect_and_calc_th();


  std::ofstream cf;  // 创建一个输出文件流对象
  // 打开文件（如果文件不存在，将创建一个新文件）
  cf.open("calibration_table.txt");
  assert(cf.is_open() && "Unable to open the file.");
  
  cf << "# genetated time: " << "datetime" << std::endl;
  cf << "# histogram number: " << "histogram_bin_num" << std::endl;
  cf << "# sample number: " << "num_samples" << std::endl;
  cf << "# tune number: " << "tune_num" << std::endl;
  cf << "# op_name    threshold    min    max" << std::endl;
  for(auto opName : infer->getAllOpNames()){
    float threshold = thresholds_map[opName];
    float min_value = std::get<0>(infer->_activations_statistics[opName]);
    float max_value = std::get<1>(infer->_activations_statistics[opName]);
    cf << opName << " " << std::setprecision(7) << threshold << " " << std::setprecision(7) << min_value << " " << std::setprecision(7) << max_value << "\n";
  }

  // 关闭文件
  cf.close();


//dump threshold of MAL3
  // Open the text file
  std::ifstream inputFile0("/home/xcgao/projects/forward-opt-cpp/build/firstarg.txt");
  if (!inputFile0){
    std::cerr << "Error opening file." << std::endl;
    return 0;
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
    return 0;
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

  auto matmul = new mlir::FORWARD::MatMul();
  std::shared_ptr<std::vector<float>> activation = std::make_shared<std::vector<float>>(589824);
  matmul->setup(arg0, arg1, nullptr, &(activation->data()[0]), 1, 1,
                768, 197, 768, false, 0, 0, 0, false,
                0, 0, 0);             

  auto phandle = (void *)matmul;
  assert(phandle != nullptr);
  auto matmul_handle = (mlir::FORWARD::MatMul *)phandle;
  matmul_handle->run();

//define min/max
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    float abs_value = 0;

    auto act_max = *(std::max_element(activation->begin(), activation->end()));
    auto act_min = *(std::min_element(activation->begin(), activation->end()));
    min_value = std::min(act_min, min_value);
    max_value = std::max(act_max, max_value);
    abs_value = std::max(abs(min_value), abs(max_value));
    if(abs_value <= 1e-5){
      // if op's outputs are all close to zero, change it to 1e-5 for them.
      min_value = -1e-5;
      max_value = 1e-5;
      abs_value = 1e-5;
      llvm::dbgs() << "[WARNING] layer " << "opname" << " is all zeros. Please check the input data correctness.\n";
    }
    // _activations_statistics[module::getName(op).str()] = std::make_tuple(min_value, max_value, abs_value);
    // abs_value = std::get<2>(_activations_statistics[name]);
    auto histogramParam = infer->histogram(activation, abs_value);
    auto hist = histogramParam.first;
    auto width = histogramParam.second;
    auto threshold = CaliMath::kl_diversity_hist(&(hist[0]), width, 2048);
    llvm::errs() << "max: " << max_value << "; min: " << min_value << "; threshold: " << threshold << "\n";

  return 0;
  
}