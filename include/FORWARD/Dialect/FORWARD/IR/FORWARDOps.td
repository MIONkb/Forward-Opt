//===-- FORWARDOps.td - FORWARD dialect operation definitions ------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines some operations of the FORWARD dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FORWARD_OPS
#define FORWARD_OPS


include "mlir/Pass/PassBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// include "FORWARD/Dialect/FORWARD/IR/FORWARDBase.td"
// #include "mlir/Dialect/Func/IR/FuncOps.h"

//===----------------------------------------------------------------------===//
// FORWARD Dialect.
//===----------------------------------------------------------------------===//

def FORWARD_Dialect : Dialect {
  let name = "FORWARD";
  let cppNamespace = "::mlir::FORWARD";

  let description = [{
    "a dialect for AI compiler-FORWARD"
  }];

  let dependentDialects = [
    // "scf::SCFDialect",
    // "::mlir::func"
  ];

  // let hasOperationAttrVerify = 1;
  // let useDefaultAttributePrinterParser = 1;
  // let usePropertiesForAttributes = 0; // newer llvm needs this 
}

//===----------------------------------------------------------------------===//
// FORWARD Dialect operations.
//===----------------------------------------------------------------------===//
class FORWARD_Op<string mnemonic, list<Trait> traits = []> :
    Op<FORWARD_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// kernel Operations for offload HW code
//===----------------------------------------------------------------------===//

def FORWARD_KernelOp : FORWARD_Op<"kernel">,
    Arguments<(ins)>,
    Results<(outs)> {
  let summary = "FORWARD KernelOp contains a kernel";

  let description = [{
    FORWARD.kernel define a region that the body of the rigion
    is a kernel which generated from a affine.for/scf.for/linalg.generic

    Syntax: //To fix

    ```
    operation ::= `soda.launch` region attr-dict?
    ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
    ```

    Example: //To fix

    ```mlir 
    soda.launch {
      "some_op"(%bx, %tx) : (index, index) -> ()
      // Assuming %val1 is defined outside the soda.launch region.
      %42 = load %val1[%bx] : memref<?xf32, 1>
    }
    ```

  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins)>,
    OpBuilder<(ins "std::string":$KernelName)>
  ];
  
  let extraClassDeclaration = [{

    /// The number of launch configuration operands, placed at the leading
    /// positions of the operand list.
    /// Legacy from copying GPU dialect.
    static constexpr unsigned kNumConfigOperands = 0;

    /// The number of region attributes containing the launch configuration,
    /// placed in the leading positions of the argument list.
    /// Legacy from copying GPU dialect.
    static constexpr unsigned kNumConfigRegionAttributes = 0;
    // unsigned kernel_Count = 0;

    static StringRef getKernelNameAttrStr() { return "KernelName"; }
    void setKernelName(std::string s) {
      StringAttr KernelNameAttr = StringAttr::get(getOperation()->getContext(),s);
      getOperation()->setAttr(getKernelNameAttrStr(), KernelNameAttr);
    }
    std::string getKernelName(){
      return getOperation()->getAttr("KernelName")
        .cast<StringAttr>().str();
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}



def FORWARD_TerminatorOp : FORWARD_Op<"terminator", [HasParent<"KernelOp">,
                                             Terminator]>,
    Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for FORWARD_KernelOp regions.";
  let description = [{
    A terminator operation for regions that appear in the body of FORWARD_KernelOp
    operation.  These regions are not expected to return any value so the
    terminator takes no operands.
  }];

  let assemblyFormat = "attr-dict";
}


//===----------------------------------------------------------------------===//
// FORWARD Attributes.
//===----------------------------------------------------------------------===//

class FORWARD_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<FORWARD_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

// A string attribute whose value are one of the values in `cases`.
class AnyStrAttrOf<list<string> cases> : StringBasedAttr<
  CPred<!foldl(
      "$_self.cast<StringAttr>().getValue() == \"" # !head(cases) # "\"",
      !foreach(case, !tail(cases),
               "$_self.cast<StringAttr>().getValue() == \"" # case # "\""),
      prev, cur, prev # " || " # cur)>,
  "string attribute whose value is " #
    !foldl(/*init*/!head(cases), /*list*/!tail(cases),
           prev, cur, prev # ", or " # cur)>;


def ArgModeAttr: AnyStrAttrOf<["ArgMin","ArgMax"]>;
def CompareModeAttr: AnyStrAttrOf<["Equal","Greater","GreaterOrEqual","Less","LessOrEqual", "NotEqual", "Not", "And"]>;
def ReduceModeAttr: AnyStrAttrOf<["ReduceMin","ReduceMax","ReduceMean","ReduceL2","ReduceL1","ReduceSum","ReduceProd"]>;
def RoiAlignModeAttr: AnyStrAttrOf<["Avg","Max"]>;
def NonZeroOrderAttr: AnyStrAttrOf<["ColMajor","RowMajor"]>;
def DetectionOutputCodeTypeAttr: AnyStrAttrOf<["CORNER", "CENTER_SIZE", "CORNER_SIZE"]>;
def YoloVersionAttr: AnyStrAttrOf<["yolov3", "yolov3_tiny", "yolov3_spp", "yolov4", "yolov5"]>;
def MatchTemplateModeAttr: AnyStrAttrOf<["TM_CCOEFF_NORMED", "TM_SQDIFF"]>;

def FORWARD_LayerGroupAttr : FORWARD_Attr<"LayerGroup", "lg"> {
  let summary = "Structure of layer group parameters";
  let parameters = (ins
    "int64_t":$out_addr,
    "int64_t":$out_size,
    "int64_t":$buffer_addr,
    "int64_t":$buffer_size,
    "bool":$eu_align,
    "DenseI64ArrayAttr":$n_idx,
    "DenseI64ArrayAttr":$n_slice,
    "DenseI64ArrayAttr":$c_idx,
    "DenseI64ArrayAttr":$c_slice,
    "DenseI64ArrayAttr":$d_idx,
    "DenseI64ArrayAttr":$d_slice,
    "DenseI64ArrayAttr":$h_idx,
    "DenseI64ArrayAttr":$h_slice,
    "DenseI64ArrayAttr":$w_idx,
    "DenseI64ArrayAttr":$w_slice,
    "int64_t":$id,
    "int64_t":$stage,
    "int64_t":$group_type
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def FORWARD_DequantMode: I32EnumAttr<"DequantMode",
    "dequant mode supported by DequantOp",
    [
      I32EnumAttrCase<"Normal", 0>,
      I32EnumAttrCase<"TFLite", 1>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::FORWARD";
}
def FORWARD_DequantModeAttr : EnumAttr<FORWARD_Dialect, FORWARD_DequantMode, "dq_mode">;

def FORWARD_RequantMode: I32EnumAttr<"RequantMode",
    "requant mode supported by RequantOp",
    [
      I32EnumAttrCase<"TFLite_LShift", 0>,
      I32EnumAttrCase<"TFLite", 1>,  // * Multi >> 31 >> shift, == QDM
      I32EnumAttrCase<"MultiplierShift", 2>, // * Multi >> shift
      I32EnumAttrCase<"OnlyShift", 3>, // >> shift
      I32EnumAttrCase<"QDM", 4>       // similar to TFLite
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::FORWARD";
}
def FORWARD_RequantModeAttr : EnumAttr<FORWARD_Dialect, FORWARD_RequantMode, "rq_mode">;

def FORWARD_RoundMode: I32EnumAttr<"RoundMode",
    "round mode supported by Round",
    [
      I32EnumAttrCase<"HalfAwayFromZero", 0>,
      I32EnumAttrCase<"HalfUp", 1>,
      I32EnumAttrCase<"HalfDown", 2>,
      I32EnumAttrCase<"HalfToEven", 3>,
      I32EnumAttrCase<"HalfToOdd", 4>,
      I32EnumAttrCase<"HalfTowardsZero", 5>,
      I32EnumAttrCase<"TowardsZero", 6>,
      I32EnumAttrCase<"Up", 7>,
      I32EnumAttrCase<"Down", 8>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::FORWARD";
}
def FORWARD_RoundModeAttr : EnumAttr<FORWARD_Dialect, FORWARD_RoundMode, "round_mode">;

def FORWARD_PoolMode: I32EnumAttr<"PoolMode",
    "pooling mode supported by PoolOp",
    [
      I32EnumAttrCase<"Avg", 0>,
      I32EnumAttrCase<"Max", 1>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::FORWARD";
}
def FORWARD_PoolModeAttr : EnumAttr<FORWARD_Dialect, FORWARD_PoolMode, "pool_mode">;

def FORWARD_LutBF16Mode : I32EnumAttr<"LutBF16Mode",
    "bf16 look up table mode",
    [
      I32EnumAttrCase<"Other", 0>,
      I32EnumAttrCase<"Mantissa", 1>,
      I32EnumAttrCase<"Slope", 2>,
      I32EnumAttrCase<"Log", 3>,
      I32EnumAttrCase<"Exp", 4>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::FORWARD";
}
def FORWARD_LutBF16ModeAttr : EnumAttr<FORWARD_Dialect, FORWARD_LutBF16Mode, "lut_mode">;

def FORWARD_ActiveMode : I32EnumAttr<"ActiveMode",
    "Activation mode for ActiveOp, for sigmoid/exp, e.g.",
    [
      I32EnumAttrCase<"TANH", 0>,
      I32EnumAttrCase<"SIGMOID", 1>,
      I32EnumAttrCase<"RELU", 2>,
      I32EnumAttrCase<"EXP", 3>,
      I32EnumAttrCase<"ELU", 4>,
      I32EnumAttrCase<"SQRT", 5>,
      I32EnumAttrCase<"SQUARE", 6>,
      I32EnumAttrCase<"RSQRT", 7>,
      I32EnumAttrCase<"ABSVAL", 8>,
      I32EnumAttrCase<"LN", 9>,
      I32EnumAttrCase<"ROUND", 10>,
      I32EnumAttrCase<"CEIL", 11>,
      I32EnumAttrCase<"FLOOR", 12>,
      I32EnumAttrCase<"SIN", 13>,
      I32EnumAttrCase<"COS", 14>,
      I32EnumAttrCase<"IS_FINITE", 15>,
      I32EnumAttrCase<"MISH", 16>,
      I32EnumAttrCase<"SWISH", 17>,
      I32EnumAttrCase<"HSWISH", 18>,
      I32EnumAttrCase<"SILU", 19>,
      I32EnumAttrCase<"ARCSIN", 20>,
      I32EnumAttrCase<"ARCCOS", 21>,
      I32EnumAttrCase<"ARCSINH", 22>,
      I32EnumAttrCase<"ARCCOSH", 23>,
      I32EnumAttrCase<"ARCTANH", 24>,
      I32EnumAttrCase<"SINH", 25>,
      I32EnumAttrCase<"COSH", 26>,
      I32EnumAttrCase<"TAN", 27>,
      I32EnumAttrCase<"SIGN", 28>,
      I32EnumAttrCase<"GELU", 29>,
      I32EnumAttrCase<"ERF", 30>,
      I32EnumAttrCase<"HSIGMOID", 31>,
      I32EnumAttrCase<"LOG_SIGMOID", 32>,
      I32EnumAttrCase<"SOFT_PLUS", 33>,
      I32EnumAttrCase<"SOFT_SIGN", 34>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::FORWARD";
}
def FORWARD_ActiveModeAttr : EnumAttr<FORWARD_Dialect, FORWARD_ActiveMode, "active_mode">;

def FORWARD_ResizeMode : I32EnumAttr<"ResizeMode",
    "Resize mode",
    [
      I32EnumAttrCase<"nearest", 0>,
      I32EnumAttrCase<"linear", 1>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::FORWARD";
}
def FORWARD_ResizeModeAttr : EnumAttr<FORWARD_Dialect, FORWARD_ResizeMode, "mode">;

def FORWARD_ResizeCoordMode : I32EnumAttr<"ResizeCoordMode",
    "Resize coord mode",
    [
      I32EnumAttrCase<"align_corners", 0>,
      I32EnumAttrCase<"half_pixel", 1>,
      I32EnumAttrCase<"pytorch_half_pixel", 2>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::FORWARD";
}
def FORWARD_ResizeCoordModeAttr : EnumAttr<FORWARD_Dialect, FORWARD_ResizeCoordMode, "coord_mode">;

def FORWARD_RunMode: I32EnumAttr<"RunMode",
    "FORWARD dialect run mode for each subnet",[
      I32EnumAttrCase<"FORWARD_STATIC",  0>,
      I32EnumAttrCase<"FORWARD_DYNAMIC", 1>,
      I32EnumAttrCase<"CPU",         2>,
      I32EnumAttrCase<"SWITCH",      3>,
      I32EnumAttrCase<"MERGE",       4>,
      I32EnumAttrCase<"UNKNOW",       5>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::FORWARD";
}
def FORWARD_RunModeAttr : EnumAttr<FORWARD_Dialect, FORWARD_RunMode, "run_mode">;

//===----------------------------------------------------------------------===//
// FORWARD Types.
//===----------------------------------------------------------------------===//

def AnyTensorOrNone: AnyTypeOf<[AnyRankedTensor, NoneType]>;


// //===----------------------------------------------------------------------===//
// // Explicit Data Block Movement Operation (Load and Store)
// //===----------------------------------------------------------------------===//
// 
// def FDRA_DataBlockLoadOp: FDRA_Op<"BlockLoad", []>{
//   let summary = "Load a part of the whole array data block to a faster memory where HW accelerator can access.";
//   let description = [{
//     This operation takes in an array memref,a header index of the data block which will be moved, and the block size.
//     Returns the new memref from the buffer given certain header index and size.
//     This operation will corporate with BlockStore and Call of KernelFunc. 
// 
//     Example: To fix !!!
//     ```mlir
//      %0 = FDRA.BlockLoad %arg1 [%arg9, 0] : memref<32x32xf32> -> memref<1x32xf32> {three_mm_32_kernel_0} 
//      %1 = FDRA.BlockLoad %arg2 [0, %arg10] : memref<32x32xf32> -> memref<32x16xf32> {three_mm_32_kernel_0}
//     ```
//   }];
//   
//   let builders = [   
//     /// Builds an affine load op with the specified map and operands.
//     // OpBuilder<(ins "AffineMap":$map, "ValueRange":$operands, "unsigned":$BlockLen)>,
//     /// Builds an affine load op with an identity map and operands.
//     // OpBuilder<(ins "Value":$DataBlock, "ValueRange":$operands, "unsigned":$BlockLen)>,
//     /// Builds a block load op with the specified map and its operands.
//     OpBuilder<(ins "Value":$DataBlock, "AffineMap":$map, "ValueRange":$operands, 
//                    "MemRefType":$resultType , "std::string":$KernelName )>,
//     // OpBuilder<(ins "Value":$DataBlock, "AffineMap":$map, "ValueRange":$operands, 
//     //                 "int64_t":$BlockLen, "std::string":$KernelName )>,
//     // /// Builds an block load op without KernelName.
//     OpBuilder<(ins "Value":$DataBlock, "AffineMap":$map, "ValueRange":$operands, 
//                    "MemRefType":$resultType)>,
//   ];
//   
// 	//Ignore Variadic<Index>:$offset at this moment
//   let arguments = (ins Arg<AnyMemRef, "the reference to load from", [MemRead]>:$OriginalMemref
//                   , Variadic<Index>:$indices);
//   
//   let results = (outs AnyStaticShapeMemRef: $result);
//   let extraClassDeclaration = [{
//     // unsigned int BlockLen = 0;
//     // void setBlockLen(unsigned l){ BlockLen = l;};
//     // unsigned int getBlockLen(){ return BlockLen;};
//     Value getOriginalMemref(){ return OriginalMemref();};
// 
//     static StringRef getMapAttrStr() { return "map"; }
//     static StringRef getKernelNameAttrStr() { return "KernelName"; }
//     // static StringRef getBlockLenAttrStr() { return "BlockLen"; }
// 
//     // int32_t getBlockLen() {
//     //   return this->getOperation()->getAttr("BlockLen")
//     //     .cast<IntegerAttr>().getValue().getSExtValue();
//     // }
//     StringRef getKernelName() {
//       Attribute KnNameAttr = this->getOperation()->getAttr("KernelName");
//       if(KnNameAttr == nullptr)
//         return "";
//       else
//         return KnNameAttr.cast<StringAttr>().strref();
//     }
//     void setKernelName(std::string s) {
//       StringAttr KernelNameAttr = StringAttr::get(getOperation()->getContext(),s);
//       getOperation()->setAttr(getKernelNameAttrStr(), KernelNameAttr);
//     }
// 
//     MemRefType getOriginalMemrefType() {
//       return OriginalMemref().getType().cast<MemRefType>();
//     }
//     MemRefType getResultType() {
//       return result().getType().cast<MemRefType>();
//     }
// 
//     operand_range getMapOperands() {
//       return {operand_begin() + 1, operand_end()};
//     }
// 
//     /// Returns the affine map used to index the memref for this operation.
//     AffineMap getAffineMap() { return getAffineMapAttr().getValue(); }
//     AffineMapAttr getAffineMapAttr() {
//       return (*this)->getAttr(getMapAttrStr()).cast<AffineMapAttr>();
//     }
// 
//   }];
// 
//   let hasCustomAssemblyFormat = 1;
//   let hasVerifier = 1;
// }
// 
// 
// def FDRA_DataBlockStoreOp: FDRA_Op<"BlockStore", []>{
//   let summary = "Store a smaller memref to a bigger memref at a specified position.";
//   let description = [{
//     This operation takes in a source array memref which comes from a BlockLoad Op. After the excution 
//     of the kernel, this array memref is changed.
//     This operation take in another target array memref and a header index of the data block pointing the place
//     the source memref will be placed.
//     No return value exists for BlockStoreOp.
//     This operation will corporate with BlockLoad and Call of KernelFunc. 
// 
//     Example: To fix !!!
//     ```mlir
//     %value = "FDRA.BlockLoad" (%buffer, %j):(memref<10xf32>, index, index) -> memref<5xf32> 
//     ```
//   }];
// 
//   let builders = [   
//     OpBuilder<(ins "Value":$SourceMemref, "Value":$TargetMemref, 
//                    "AffineMap":$map, "ValueRange":$operands, 
//                    "std::string":$KernelName )>,
//     // /// Builds an block load op without KernelName.
//     OpBuilder<(ins "Value":$SourceMemref, "Value":$TargetMemref, 
//                    "AffineMap":$map, "ValueRange":$operands)>,
//   ];
// 
// 	//Ignore Variadic<Index>:$offset at this moment
//   let arguments = (ins Arg<AnyMemRef, "A memref from a BlockLoad Op", [MemWrite]>:$SourceMemref
//                   , Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$TargetMemref
//                   , Variadic<Index>:$indices);
//   // let results = (outs AnyStaticShapeMemRef: $result); /// No results for 
//   let extraClassDeclaration = [{
//     // unsigned int BlockLen = 0;
//     // void setBlockLen(unsigned l){ BlockLen = l;};
//     // unsigned int getBlockLen(){ return BlockLen;};
//     Value getSourceMemref(){ return SourceMemref();};    
//     Value getTargetMemref(){ return TargetMemref();};
// 
//     static StringRef getMapAttrStr() { return "map"; }
//     static StringRef getKernelNameAttrStr() { return "KernelName"; }
//     // static StringRef getBlockLenAttrStr() { return "BlockLen"; }
// 
//     // int32_t getBlockLen() {
//     //   return this->getOperation()->getAttr("BlockLen")
//     //     .cast<IntegerAttr>().getValue().getSExtValue();
//     // }
//     StringRef getKernelName() {
//       return this->getOperation()->getAttr("KernelName")
//         .cast<StringAttr>().strref();
//     }
//     void setKernelName(std::string s) {
//       StringAttr KernelNameAttr = StringAttr::get(getOperation()->getContext(),s);
//       getOperation()->setAttr(getKernelNameAttrStr(), KernelNameAttr);
//     }
// 
//     MemRefType getSourceMemrefType() {
//       return SourceMemref().getType().cast<MemRefType>();
//     }
//     MemRefType getTargetMemrefType() {
//       return TargetMemref().getType().cast<MemRefType>();
//     }
// 
//     operand_range getMapOperands() {
//       return {operand_begin() + 2, operand_end()}; // + 2 or + 1?
//     }
// 
//   }];
// 
//   let hasCustomAssemblyFormat = 1;
//   let hasVerifier = 1;
// }
// 
// 
// 
// //===----------------------------------------------------------------------===//
// // KernelCallOp
// //===----------------------------------------------------------------------===//
// 
// def KernelCallOp : FDRA_Op<"KernelCall",
//     [CallOpInterface, MemRefsNormalizable,
//      DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
//   let summary = "call operation(different from call in func)";
//   let description = [{
//     The `func.call` operation represents a direct call to a function that is
//     within the same symbol scope as the call. The operands and result types of
//     the call must match the specified function type. The callee is encoded as a
//     symbol reference attribute named "callee".
// 
//     Example:
// 
//     ```mlir
//     %2 = fdra.kernelcall @my_add(%0, %1) : (f32, f32) -> f32
//     ```
//   }];
// 
//   let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
//   let results = (outs Variadic<AnyType>);
// 
//   let builders = [
//     OpBuilder<(ins "func::FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
//       $_state.addOperands(operands);
//       $_state.addAttribute("callee", SymbolRefAttr::get(callee));
//       $_state.addTypes(callee.getFunctionType().getResults());
//     }]>,
//     OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
//       CArg<"ValueRange", "{}">:$operands), [{
//       $_state.addOperands(operands);
//       $_state.addAttribute("callee", callee);
//       $_state.addTypes(results);
//     }]>,
//     OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
//       CArg<"ValueRange", "{}">:$operands), [{
//       build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
//     }]>,
//     OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
//       CArg<"ValueRange", "{}">:$operands), [{
//       build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
//             results, operands);
//     }]>];
// 
//   let extraClassDeclaration = [{
//     FunctionType getCalleeType();
// 
//     /// Get the argument operands to the called function.
//     operand_range getArgOperands() {
//       return {arg_operand_begin(), arg_operand_end()};
//     }
// 
//     operand_iterator arg_operand_begin() { return operand_begin(); }
//     operand_iterator arg_operand_end() { return operand_end(); }
// 
//     /// Return the callee of this operation.
//     CallInterfaceCallable getCallableForCallee() {
//       return (*this)->getAttrOfType<SymbolRefAttr>("callee");
//     }
//   }];
// 
//   let assemblyFormat = [{
//     $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
//   }];
// }
//===----------------------------------------------------------------------===//
// Outlined Func Operations for offload HW code
//===----------------------------------------------------------------------===//

// def FDRA_FDRAFuncOp : FDRA_Op<"func", [//HasParent<"SODAModuleOp">,
//                                     AutomaticAllocationScope, FunctionOpInterface,
//                                     IsolatedFromAbove, Symbol]> {
//   let summary = "Function to outline a FDRA.Kernel";

//   let description = [{
//     Defines a function to be outlined from a FDRA.Kernel

//     // SODA functions are either kernels (as indicated by the `kernel` attribute) or
//     // regular functions. The former can be launched from the host side, while the
//     // latter are device side only.

//     // The memory attribution defines SSA values that correspond to memory buffers
//     // allocated in the memory hierarchy of the SODA (see below).

//     // The operation has one attached region that corresponds to the body of the
//     // function. The region arguments consist of the function arguments without
//     // modification, followed by buffers defined in memory annotations.
//     // Declarations of SODA functions, i.e. not having the
//     // body region, are not supported.

//     Syntax:

//     ```
//     op ::= `soda.func` symbol-ref-id `(` argument-list `)` (`->`
//     function-result-list)?
//            memory-attribution `kernel`? function-attributes? region

//     memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
//                            (`private` `(` ssa-id-and-type-list `)`)?
//     ```

//     Example:

//     ```mlir
//     soda.func @foo(%arg0: index)
//         workgroup(%workgroup: memref<32xf32, 3>)
//         private(%private: memref<1xf32, 5>)
//         kernel
//         attributes {qux: "quux"} {
//       soda.return
//     }
//     ```

//     The generic form illustrates the concept

//     ```mlir
//     "soda.func"(%arg: index) {sym_name: "foo", kernel, qux: "quux"} ({
//     ^bb0(%arg0: index, %workgroup: memref<32xf32, 3>,
//          %private: memref<1xf32, 5>):
//       "soda.return"() : () -> ()
//     }) : (index) -> ()
//     ```

//     Note the non-default memory spaces used in memref types in memory
//     attribution.
//   }];

//   let arguments = (ins TypeAttrOf<FunctionType>:$function_type);
//   let regions = (region AnyRegion:$body);

//   let skipDefaultBuilders = 1;

//   let builders = [
//     OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
//       CArg<"TypeRange", "{}">:$workgroupAttributions,
//       CArg<"TypeRange", "{}">:$privateAttributions,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
//   ];

//   let extraClassDeclaration = [{
//     /// Returns `true` if the SODA function defined by this Op is a kernel, i.e.
//     /// it is intended to be launched from host.
//     bool isKernel() {
//       return (*this)->getAttrOfType<UnitAttr>(SODADialect::getKernelFuncAttrName()) !=
//              nullptr;
//     }

//     /// Change the type of this function in place. This is an extremely
//     /// dangerous operation and it is up to the caller to ensure that this is
//     /// legal for this function, and to restore invariants:
//     ///  - the entry block args must be updated to match the function params.
//     ///  - the argument/result attributes may need an update: if the new type
//     ///  has less parameters we drop the extra attributes, if there are more
//     ///  parameters they won't have any attributes.
//     // TODO: consider removing this function thanks to rewrite patterns.
//     void setType(FunctionType newType);

//     /// Returns the number of buffers located in the workgroup memory.
//     unsigned getNumWorkgroupAttributions() {
//       return (*this)->getAttrOfType<IntegerAttr>(getNumWorkgroupAttributionsAttrName())
//           .getInt();
//     }

//     /// Returns a list of block arguments that correspond to buffers located in
//     /// the workgroup memory
//     ArrayRef<BlockArgument> getWorkgroupAttributions() {
//       auto begin =
//           std::next(getBody().args_begin(), getFunctionType().getNumInputs());
//       auto end = std::next(begin, getNumWorkgroupAttributions());
//       return {begin, end};
//     }

//     /// Adds a new block argument that corresponds to buffers located in
//     /// workgroup memory.
//     BlockArgument addWorkgroupAttribution(Type type, Location loc);

//     /// Returns the number of buffers located in the private memory.
//     unsigned getNumPrivateAttributions() {
//       return getBody().getNumArguments() - getFunctionType().getNumInputs() -
//           getNumWorkgroupAttributions();
//     }

//     /// Returns a list of block arguments that correspond to buffers located in
//     /// the private memory.
//     ArrayRef<BlockArgument> getPrivateAttributions() {
//       // Buffers on the private memory always come after buffers on the workgroup
//       // memory.
//       auto begin =
//           std::next(getBody().args_begin(),
//                     getFunctionType().getNumInputs() + getNumWorkgroupAttributions());
//       return {begin, getBody().args_end()};
//     }

//     /// Adds a new block argument that corresponds to buffers located in
//     /// private memory.
//     BlockArgument addPrivateAttribution(Type type, Location loc);

//     /// Returns the name of the attribute containing the number of buffers
//     /// located in the workgroup memory.
//     static StringRef getNumWorkgroupAttributionsAttrName() {
//       return "workgroup_attributions";
//     }

//     /// Returns the type of this function.
//     /// FIXME: We should drive this via the ODS `type` param.
//     FunctionType getFunctionType() { return function_type(); }

//     /// Returns the argument types of this function.
//     ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

//     /// Returns the result types of this function.
//     ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

//     // /// Hooks for the input/output type enumeration in FunctionLike .
//     // unsigned getNumFuncArguments() { return getFunctionType().getNumInputs(); }
//     // unsigned getNumFuncResults() { return getFunctionType().getNumResults(); }

//     /// Returns the keywords used in the custom syntax for this Op.
//     static StringRef getWorkgroupKeyword() { return "workgroup"; }
//     static StringRef getPrivateKeyword() { return "private"; }
//     static StringRef getKernelKeyword() { return "kernel"; }

//     /// Hook for FunctionOpInterface verifier.
//     LogicalResult verifyType();

//     /// Verifies the body of the function.
//     LogicalResult verifyBody();
//   }];

//   let hasCustomAssemblyFormat = 1;
// }




// // We define a toy operation by inheriting from our base 'Toy_Op' class above.
// // Here we provide the mnemonic and a list of traits for the operation. The
// // constant operation is marked as 'NoSideEffect' as it is a pure operation
// // and may be removed if dead.
// def MulAccum : MyTestOP<"MulAccum", [NoSideEffect]> {
//   // Provide a summary and description for this operation. This can be used to
//   // auto-generate documentation of the operations within our dialect.
//   let summary = "Multiply and Accumulate";
//   let description = [{"
//     MulAccum(MAC) operation is a combination of Multiply and accumulate which
//     are common in Digital-Signal-Processing. MulAccum takes in two multipilers
//     and one sum-up number, and output a MAC result.
//     For example:

//     ```mlir
//       %4 = MyTest.MulAccum %1 MulF %2 AddF %0: f32
//     ```
//     "
//   }];

//   // The MulAccum operation takes 3 attribute as the only input.
//   let arguments = (ins F32:$MUL1, F32:$MUL2,  F32:$ADDEND);

//   // The MulAccum operation returns a single value of TensorType.
//   let results = (outs F32:$Result);

//   // Indicate that the operation has a custom parser and printer method.
// //   let hasCustomAssemblyFormat = 1;
//   let assemblyFormat = [{$MUL1` ``MulF` ` `$MUL2` ` `AddF` ` `$ADDEND` ``:`` `type($Result) attr-dict}] ;
//   // Allow building an MulAccum with from the 3 input operands.
//   // let builders = [ ];

//   // Indicate that additional verification for this operation is necessary.
//   let hasCanonicalizer = 1;
//   let hasVerifier = 0;
// }

#endif // FORWARD_OPS
