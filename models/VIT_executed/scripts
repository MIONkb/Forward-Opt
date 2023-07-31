
##
# TORCH.mlir and LINALG_TENSOR.mlir are generated from torchscript_linear.py
source ../../env.sh ### your path

#######################
# lower torch to tosa
#######################
torch-mlir-opt \
  --torch-simplification-pipeline \
  --torch-shape-refinement-pipeline\
  --torch-func-backend-type-conversion\
  --convert-torch-to-tosa \
  --torch-simplify-dtype-calculations \
  --torch-simplify-shape-calculations \
  --torch-finalizing-backend-type-conversion \
  --canonicalize --allow-unregistered-dialect\
  TORCH.mlir -o tosa.mlir 
 # --mlir-print-ir-after-all 2>&1 | cat > before_tosa_intermediate.mlir 

mlir-opt \
 --tosa-infer-shapes --tosa-validate --tosa-layerwise-constant-fold \
 --tosa-optional-decompositions --canonicalize --convert-elementwise-to-linalg\
  tosa.mlir -o tosa_opt.mlir 
 # --mlir-print-ir-after-all 2>&1 | cat > before_tosa_intermediate.mlir 

#######################
# If run quatization for this mlir
#######################
seperate_weights.py \
    --mlir tosa_opt.mlir \
    --modelname vit \
    --thread 1
torch-mlir-opt --mlir-elide-elementsattrs-if-larger=4 --canonicalize vit_simplified.mlir -o tosa_elided.mlir

run_calibration.py tosa_elided.mlir --dataset ../dataset/Cat/ \
 --weight_npz vit_origin_weight.npz

forward-opt tosa_elided.mlir\
 --forward-import-calibration-table \
 --forward-transform-to-quantized \
 -o tosa_quantized.mlir

mv tosa_quantized.mlir tosa_opt.mlir 

#######################
# quantization ned
#######################

#######################
# lower tosa to linalg: this process needs a newer version of mlir-opt than:
# https://github.com/llvm/llvm-project/tree/4553dc46a05ec6f1e2aebcde1ce185772a26780b
#######################
forward-opt  \
  --pass-pipeline="builtin.module(func.func(tosa-to-tensor,  tosa-to-arith, tosa-to-linalg-named, tosa-to-linalg))"\
  tosa_opt.mlir  -o linalg_tensor.mlir  \
  --mlir-print-ir-after-all 2>&1 | cat > before_linalg_intermediate.mlir 


#######################
# lower linalg
#######################
mlir-opt \
 --promote-buffers-to-stack --linalg-bufferize --empty-tensor-to-alloc-tensor  \
 --tensor-bufferize --func-bufferize --arith-bufferize --bufferization-bufferize \
 --convert-linalg-to-affine-loops  --promote-buffers-to-stack --canonicalize\
 linalg_tensor.mlir > affine.mlir

mlir-opt --arith-expand --memref-expand \
 -normalize-memrefs  --affine-simplify-structures \
 --affine-loop-fusion  --cse\
 -lower-affine --scf-for-loop-canonicalization  -convert-scf-to-cf\
 --convert-math-to-llvm --convert-math-to-libm \
 --convert-arith-to-llvm \
 -convert-func-to-llvm=use-bare-ptr-memref-call-conv \
 -convert-memref-to-llvm --reconcile-unrealized-casts \
 affine.mlir -o llvm.mlir \
 --mlir-print-ir-after-all 2>&1 | cat > intermediate_llvm.mlir


#######################
# lower linalg to affine
#######################
mlir-opt  --canonicalize \
  -linalg-fuse-elementwise-ops --convert-tensor-to-linalg  -empty-tensor-to-alloc-tensor \
  --eliminate-empty-tensors  --cse --canonicalize -linalg-bufferize -arith-bufferize \
  -tensor-bufferize -func-bufferize  -finalizing-bufferize -buffer-deallocation \
  -buffer-results-to-out-params --cse --canonicalize -linalg-generalize-named-ops \
  -convert-linalg-to-affine-loops -fold-memref-alias-ops --canonicalize \
  linalg_tensor.mlir -o affine.mlir 
  
#######################
# lower LLVM IR
#######################
mlir-opt   --erase-memory-copy --affine-simplify-structures --canonicalize   \
  --affine-simplify-structures --forward-memref-allocations  --lower-affine \
  --canonicalize --cse   | ./thirdparty/Polygeist/llvm-project/build/bin/mlir-opt --lower-affine   \
  -convert-vector-to-scf -convert-linalg-to-loops   --lower-affine --convert-scf-to-cf \
  --canonicalize --cse -convert-linalg-to-llvm -convert-vector-to-llvm --convert-math-to-llvm  \
  --convert-math-to-libm --lower-affine --arith-expand -convert-arith-to-llvm --memref-expand \
  --convert-memref-to-llvm --convert-func-to-llvm="use-bare-ptr-memref-call-conv=1" --convert-index-to-llvm  \
  -reconcile-unrealized-casts  --cse |  ./thirdparty/Polygeist/bin/polygeist-opt --lower-affine  \
  --convert-polygeist-to-llvm="use-c-style-memref=1" --canonicalize  \
 affine.mlir -o llvm.mlir

mlir-translate llvm.mlir --mlir-to-llvmir -opaque-pointers=0 -o forward.ll

  
#######################
# compile
#######################

llc forward.ll  --filetype=obj -o forward.o
g++  -o main test_vit.cpp  forward.o  -lm