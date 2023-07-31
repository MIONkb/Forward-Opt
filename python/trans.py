#!/usr/bin/env python3
import pymlir
import mlir.ir



import mlir.dialects



from utils.mlir_parser import MlirParser


if __name__ == '__main__':


    print("before mlir.ir.Module.parse:")

    parser = MlirParser("tosa_elided.mlir")
    module = parser.module

    print("before pymlir.py_module.load")
    md = pymlir.py_module()
    md.load("tosa_elided.mlir")


    # moduleobj = module._CAPIPtr
    # print("module:", type(moduleobj))


