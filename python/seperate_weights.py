#!/usr/bin/env python3
# from transform.MLIRImporter import MLIRImporter, Platform
from transform.BaseConverter import BaseConverter
from utils.mlir_parser import MlirParser, Operation
from tqdm import tqdm
import numpy as np

## multi threads
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import mlir
# from mlir.ir import *

import argparse
# import pymlir


class WeightsSeperator(BaseConverter):
    Tensor_op_type_list = [
        ### conv
        "linalg.conv_1d", "linalg.conv_2d", "linalg.conv_3d",

        "linalg.conv_1d_nwc_wcf", "linalg.conv_1d_ncw_fcw",
        "linalg.conv_2d_nhwc_hwcf", "linalg.conv_2d_nhwc_fhwc",
        "linalg.conv_2d_nchw_fchw", "linalg.conv_2d_nhwc_hwcf_q", 
        "linalg.conv_2d_nchw_fchw", "linalg.conv_2d_ngchw_fgchw",
        "linalg.conv_3d_ndhwc_dhwcf", "linalg.conv_3d_ndhwc_dhwcf_q", 
        
        "linalg.depthwise_conv_1d_nwc_wc","linalg.depthwise_conv_1d_nwc_wcm", 
        "linalg.depthwise_conv_2d_nhwc_hwc", "linalg.depthwise_conv_2d_nhwc_hwc_q",
        "linalg.depthwise_conv_2d_nhwc_hwcm", "linalg.depthwise_conv_2d_nhwc_hwcm_q",
        "linalg.depthwise_conv_2d_nchw_chw", 
        "linalg.depthwise_conv_3d_ndhwc_dhwc", "linalg.depthwise_conv_3d_ndhwc_dhwcm",
        

        ### matmul 
        "linalg.batch_matmul", "linalg.matmul", "linalg.matmul_unsigned",
        "linalg.quantized_matmul", "linalg.quantized_batch_matmul" , 
        "linalg.batch_matmul_transpose_b", "linalg.batch_reduce_matmul",
        "linalg.matmul_transpose_b",

        ### Vector matrix
        "linalg.matvec", "linalg.vecmat","linalg.batch_matvec",

        ### transpose
        "linalg.transpose", "linalg.mmt4d", 

        ### pooling
        "linalg.pooling_nhwc_sum", "linalg.pooling_nchw_sum",
        "linalg.pooling_nhwc_max", "linalg.pooling_nhwc_max_unsigned",
        "linalg.pooling_nchw_max", "linalg.pooling_nhwc_min",
        "linalg.pooling_nhwc_min_unsigned", "linalg.pooling_nwc_sum",
        "linalg.pooling_ncw_sum", "linalg.pooling_nwc_max",
        "linalg.pooling_nwc_max_unsigned", "linalg.pooling_ncw_max",
        "linalg.pooling_nwc_min", "linalg.pooling_nwc_min_unsigned",
        "linalg.pooling_ndhwc_sum", "linalg.pooling_ndhwc_max",
        "linalg.pooling_ndhwc_min",

        "linalg.generic",

        ### fill
        "linalg.fill", "linalg.fill_rng_2d",

        "linalg.copy",
        "linalg.dot",
        # """ This list is obtained from 
        #     /mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yaml
        #     /mlir/Dialect/Linalg/IR/LinalgOps.td
        #     /mlir/Dialect/Linalg/IR/LinalgStructuredOps.td
        #     ops in "/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td" is not included.
        #     TODO:Check this later!!
        # """

        ### To be determined
        #"linalg.elemwise_unary", "linalg.elemwise_binary"


        ### for tosa
        "tosa.argmax", 
        "tosa.avg_pool2d", "tosa.max_pool2d", 

        "tosa.conv2d", "tosa.conv3d", 
        "tosa.depthwise_conv2d", "tosa.transpose_conv2d", 
        "tosa.fully_connected", "tosa.matmul", 

        "tosa.rfft2d", "tosa.clamp", 
        "tosa.sigmoid", "tosa.tanh", 
        "tosa.rsqrt", "tosa.select",
        "tosa.greater", "tosa.greater_equal", 
        "tosa.reduce_all", "tosa.reduce_any", 

        "tosa.add", "tosa.div", "tosa.mul", "tosa.pow",
        "tosa.sub", "tosa.abs", "tosa.clz", "tosa.ceil",
        "tosa.exp", "tosa.floor","tosa.log", "tosa.negate",
        "tosa.table", "tosa.reciprocal",

        "tosa.arithmetic_right_shift", 
        
        "tosa.bitwise_and", "tosa.bitwise_or", 
        "tosa.bitwise_xor", "tosa.logical_and",
        "tosa.bitwise_not", 

        "tosa.logical_left_shift", 
        "tosa.logical_right_shift",
        "tosa.logical_or", 
        "tosa.logical_xor",
        "tosa.logical_not",

        # "tosa.maximum", "tosa.minimum",

        "tosa.reduce_max", "tosa.reduce_min",
        "tosa.reduce_prod", "tosa.reduce_sum",

        "tosa.concat", "tosa.pad","tosa.reshape", "tosa.reverse",
        "tosa.slice", "tosa.tile","tosa.transpose", "tosa.gather",
        "tosa.scatter", "tosa.resize","tosa.cast", "tosa.rescale",
        "tosa.identity", "tosa.custom",
        
        "tosa.cond_if", "tosa.while_loop",
    ] 


    def __init__(self,
                 mlir_file,
                 model_name:str,
                 output_name,
                 thread_num:int):
        super().__init__()
        self.model_name = model_name
        self.weight_file = "{}_origin_weight.npz".format(model_name)
        self.mlir = mlir_file
        self.output_mlir = output_name + ".mlir"
        self.debug_mlir = output_name + "_dbug.mlir"
        self.thread_num = int(thread_num)
        self.manager = mp.Manager

        """ Get all weights op(like arith.constant) """
        self.mlir_parser = MlirParser(self.mlir)
        self.body = self.mlir_parser.body
        self.ops = self.mlir_parser.ops
        print("[Info] Mlir Parsing finished.")
        self.tensor_store_ops = self.find_all_weights(self.body)
        print("[Info] Get all Number of weights op:{}".format(len(self.tensor_store_ops)))

        """ Add an attribute "idx" to every op """
        idx = 0
        for op in self.ops:
            op.op.attributes["idx"] = mlir.ir.StringAttr.get(str(idx), self.mlir_parser.ctx)
            idx += 1
        """ Get all weights npz """
        self.weights_dic = self.generate_weights_dic(self.tensor_store_ops)
        self.WeightToNpz(weight_file = self.weight_file, weights_dic = self.weights_dic)
        print("[Info] Save float weight npz: {}".format(self.weight_file))

        """ Erase weights in MLIR file """
        self.DelWeightsValueAndGetLoc(self.tensor_store_ops)

        mlir_format = self.mlir_parser.module.operation.get_asm(enable_debug_info=False)
        # mlir_txt = self.mlir_parser.module.context
        with open(self.output_mlir, "w") as f:
            f.write(mlir_format)
        print("[Info] Generate new mlir without weight values: {}".format(self.output_mlir))

        mlir_format = self.mlir_parser.module.operation.get_asm(enable_debug_info=True)
        # mlir_txt = self.mlir_parser.module.context
        with open(self.debug_mlir, "w") as f:
            f.write(mlir_format)
        print("[Info] Generate new mlir with debug info: {}".format(self.debug_mlir))


    def __del__(self):
        if self.mlir != None:
            del self.mlir
            self.mlir = None

    # def _find_weight(op_name, all_ops_list, all_op_name_list, fp_store_ops):
    def _find_weight(self, op):
        """ Judge whether an op stores fp weights or bias with single thread """
        users_type = []
        # if isinstance(op, mlir.dialects._arith_ops_ext.ConstantOp):
        mlir_op = op.op
        if str(op.type) == "arith.constant":
            print(".",end="")
            # print("op:", op,",type:", mlir_op.type)
            if str(mlir_op.type) in ["i64", "i32", "i16", "i8", "index"]:
                # print("op:", op,",type:", mlir_op.type)
                return
            users_type = MlirParser.get_users_type_by_op_name(op_name = op.name
                                                , all_ops_list = self.ops
                                                , all_op_name_list = self.all_op_name_list)
            # print("here2")
            for usr_type in users_type:
                if usr_type in WeightsSeperator.Tensor_op_type_list:
                    self.fp_store_ops.append(op) ## list.append meets Thread-safty in python 
                    return

        elif str(op.type) == "tosa.const":
            print(".",end="")
            v = mlir_op.value
            # print("op:", op,",type:", type(mlir_op), "v:", v)
            # if str(mlir_op.type) in ["i64", "i32", "i16", "i8", "index"]:
            #     # print("op:", op,",type:", mlir_op.type)
            #     return
            users_type = MlirParser.get_users_type_by_op_name(op_name = op.name
                                                , all_ops_list = self.ops
                                                , all_op_name_list = self.all_op_name_list)
            # print("here2")
            for usr_type in users_type:
                if usr_type in WeightsSeperator.Tensor_op_type_list:
                    self.fp_store_ops.append(op) ## list.append meets Thread-safty in python 
                    return

    @staticmethod
    def err_call_back(err):
        print(f'Multi-thread error：{str(err)}')

    def find_all_weights(self, func_body):
        """ Get all weights from mlir on linalg/tensor dialect using multi-thread """
        self.fp_store_ops = []
        self.all_op_name_list = [op.name for op in self.ops]
        with ThreadPoolExecutor(max_workers=self.thread_num) as executor :
            result=list(tqdm(executor.map(self._find_weight, self.ops), total=len(self.ops)))

        print()
        return self.fp_store_ops
        # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        #     [executor.submit(self._find_weight, op) for op in self.ops]
            # 使用 as_completed() 获取已经完成的函数的返回值
            # for future in concurrent.futures.as_completed(futures):
            #     result = future.result()
            #     results.append(result)

        # for i in tqdm.trange(len(func_body.operations)):
        #     pool.apply_async(WeightsSeperator._find_weight, 
        #                      (self.ops,),
        #                     # (Operation.name(func_body.operations[i]),self.ops,),#self.all_op_name_list, self.fp_store_ops),
        #                     error_callback=WeightsSeperator.err_call_back)

        # pool.close()
        # pool.join()
     
            # op = func_body.operations[i]
            # # print("op.type:", op.type)
            # users_type = []

            # if isinstance(op, mlir.dialects._arith_ops_ext.ConstantOp):
            #     # print("op.type:", op.type)
            #     if str(op.type) in ["i64", "i32", "i16", "i8", 'index']:
            #         # print("int type")
            #         continue

            #     for usr_type in self.mlir_parser.get_users_type_by_op_name(Operation.name(op)):
            #         # print("    usr_type:", usr_type)
            #         users_type.append(usr_type) 
            #     print("here2")
            #     for usr_type in users_type:
            #         if usr_type in WeightsSeperator.Tensor_op_type_list:
            #             tensor_store_ops.append(op)
            #             break
            #     print("here3")


    
    def generate_weights_dic(self, weights_ops):
        """ extract weights from constant ops and generate .npz file """
        weights_dic = {}
        constop_cnt = 0

        for operation in weights_ops:
            op = operation.op
            # print("------------------------------")
            weights_Attr = op.attributes["value"]
            shape = Operation.shape(op)
            name =  Operation.name(op)
            weights_value = np.array([])
            # print(name,end=",")
            # print("op:", op)
            # print("op:", op.literal_value)
            # print("weights_Attr:", type(weights_Attr))
            if str(operation.type) == "arith.constant": ## For linalg dialect
                idx =  mlir.ir.StringAttr(op.attributes["idx"])
                npz_loc = str(idx) + ".arith.constant"
                if str(op.type) in ["tensor<f64>", "tensor<f32>"]:
                    weights_FPElemAttr = mlir.ir.DenseFPElementsAttr(weights_Attr)
                    weights_value = np.array([weights_FPElemAttr[idx] for idx in range(len(weights_FPElemAttr))]
                                                    ,dtype=np.float32)
                    weights_dic[npz_loc] = weights_value
            
                elif shape == []: ## a single float value
                    weight_FP = mlir.ir.FloatAttr(weights_Attr).value
                    weights_value = np.array([weight_FP],dtype=np.float32)
                    # print("FloatAttr:", weights_value)
                    weights_dic[npz_loc] = weights_value

                else : ## tensor
                    weights_FPElemAttr = mlir.ir.DenseFPElementsAttr(weights_Attr)
                    weights_value = np.array([weights_FPElemAttr[idx] for idx in range(len(weights_FPElemAttr))]
                                                ,dtype=np.float32)
                    tensor = weights_value.reshape(shape)
                    # print("shape:", shape)
                    # print("tensor:", tensor)
       
                    weights_dic[npz_loc] = tensor
                op.attributes["npz_loc"] = mlir.ir.StringAttr.get(npz_loc, self.mlir_parser.ctx)
                constop_cnt += 1

            elif str(operation.type) == "tosa.const": ## For TOSA dialec
                if isinstance(weights_Attr, mlir.ir.DenseFPElementsAttr):
                    weights_FPElemAttr = mlir.ir.DenseFPElementsAttr(weights_Attr)
                    weights_value = np.array([weights_FPElemAttr[idx] for idx in range(len(weights_FPElemAttr))]
                                                ,dtype=np.float32)
                elif isinstance(weights_Attr, mlir.ir.DenseIntElementsAttr):
                    weights_I32ElemAttr = mlir.ir.DenseIntElementsAttr(weights_Attr)
                    weights_value = np.array([weights_I32ElemAttr[idx] for idx in range(len(weights_I32ElemAttr))]
                                                ,dtype=np.float32)
                tensor = weights_value.reshape(shape)
                # print("shape:", shape)
                 # print("tensor:", tensor)
                idx =  mlir.ir.StringAttr(op.attributes["idx"])
                npz_loc = idx.value + ".tosa.const"
                print("npz_loc:", npz_loc)
                op.attributes["npz_loc"] = mlir.ir.StringAttr.get(npz_loc, self.mlir_parser.ctx)
                weights_dic[npz_loc] = tensor
                constop_cnt += 1
                
            

        return weights_dic

    def DelWeightsValueAndGetLoc(self, weights_ops):
        """ erase weights value from constant ops """
        for operation in weights_ops:
            op = operation.op
            print("------------------------------")
            print("before op:", op)
            print("value:", type(op.value))
            
            # del op.attributes["value"]
            # op.attributes["value"]=mlir.ir.DenseFPElementsAttr()
            # print("after del op:", op)
            
    
    def WeightToNpz(self, weight_file, weights_dic):
        """ Overide WeightToNpz() in BaseConverter class """
        # tensor_npz = {}
        # for name in self.tensors:
        #     tensor_npz[name] = self.tensors[name]
        np.savez(weight_file, **weights_dic)

    def get_loc(self, names):
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=self.mlir.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=self.mlir.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))



if __name__ == '__main__':
    print("[Info] Starting seperating weights from mlir file...")
    # print("MLIR Forward Toolchain {}".format(pymlir.module().version))
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--mlir", required=True,
                        help="a linalg mlir fp32 model")
    parser.add_argument("--modelname", required=True, help='name of model')

    parser.add_argument("--thread", required=False, help='number of threads to run')

    args = parser.parse_args()
    
    # with open(args.mlir, 'r') as f:
    #     context = f.read()
    # print("top context:")
    # print(context)
    # ctx = mlir.ir.Context()
    # ctx.allow_unregistered_dialects = True
    # module = mlir.ir.Module.parse(context, ctx)

    thread_num = args.thread if args.thread else 1 

    tool = WeightsSeperator(mlir_file = args.mlir, model_name = args.modelname
                            ,output_name = "{}_simplified".format(args.modelname)
                            ,thread_num = thread_num)
                        