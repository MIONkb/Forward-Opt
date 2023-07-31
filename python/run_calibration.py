#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import re
import argparse
import mlir
# from mlir.ir import *
# from utils.mlir_parser import MlirParser
import pymlir

# from calibration.kld_calibrator import ActivationCalibrator, ActivationCalibrator2
from calibration.kld_calibrator import ActivationCalibrator2
from calibration.data_selector import DataSelector


if __name__ == '__main__':
    #print("SOPHGO Toolchain {}".format(pymlir.module().version))

    print("Forward-OPT Start")
    # yapf: disable

    parser = argparse.ArgumentParser()
    parser.add_argument('mlir_file', metavar='mlir_file', help='mlir file')
    parser.add_argument('--dataset', type=str, help='dataset for calibration')
    parser.add_argument('--data_list', type=str, help='Input list file contain all input')
    parser.add_argument('--input_num', type=int, default=0, help='num of images for calibration')
    parser.add_argument('--tune_num', type=int, default=5, help='num of images for tune')
    parser.add_argument('--histogram_bin_num', type=int, default=2048,
                        help='Specify histogram bin numer for kld calculate')
    parser.add_argument('--weight_npz', type=str, help='weight npz of mlir')
    parser.add_argument('-o', '--calibration_table', type=str, help='output threshold table')
    parser.add_argument('--debug_cmd', type=str, default='', help='debug cmd')
    # yapf: enable
    args = parser.parse_args()


#     mlirparser = MlirParser(args.mlir_file)

#     md = pymlir.py_module()
#     md.load(args.mlir_file)

    selector = DataSelector(args.dataset, args.input_num, args.data_list)
    # calibration
    # if 'use_old_cali' in args.debug_cmd:
    #     calibrator = ActivationCalibrator(args, selector)
    # else:
    #     calibrator = ActivationCalibrator2(args, selector)
#     with open(args.mlir_file, 'r') as f:
#             context = f.read()
#     print("top context:")
#     print(context)
#     ctx = mlir.ir.Context()
#     ctx.allow_unregistered_dialects = True
#     module = mlir.ir.Module.parse(context, ctx)
#     print("top module:",module)
#     print("before md")
#     md = pymlir.py_module(args.mlir_file)
#     pymlir.load_py_module(args.mlir_file)
#     print("md")
    module = pymlir.py_module()
    module.set_weight_npz(str(args.weight_npz))
    module.load(args.mlir_file)
    # print("here",module)
    
    calibrator = ActivationCalibrator2(args, selector, py_module=module)
    calibrator.run()
