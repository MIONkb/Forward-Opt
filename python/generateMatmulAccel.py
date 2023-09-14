#!/usr/bin/python

import sys
import re

def codeGenHeader():
    print("""
/*
 * ======================================================================
 * MatmulAccelRuntime.cpp
 * ======================================================================
 * This file includes the interfaces to compile and call matmul accelerator.
 * Don't change any codes except the main() function.  
 *
*/
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <stdio.h>
#include <iostream>

#include \"mmsim.h\"
#include \"apimm.h\"
 """)

def codeGenGenericDeclares(declares):
    print("extern \"C\" {")
    for dec in declares:
        print("\t" + dec.toCString())
    print("}")
    print()

def codeGenGlobalDeclares(line):
    funcName = line.split("@")[1].split("(")[0]
    match = re.match(func_pattern, funcName)

    matmul_accel = match.group(1) 
    idx = match.group(2)  
    pattern_attributes = r'attributes {(.*?)}'
    match_attributes = re.search(pattern_attributes, line)

    pattern_input0 = r'input0\s*=\s*dense<\[([0-9,\s]+)'
    pattern_input1 = r'input1\s*=\s*dense<\[([0-9,\s]+)\]>'
    pattern_output = r'output\s*=\s*dense<\[([0-9,\s]+)\]>'
    pattern_m = r'm\s*=\s*([0-9]+)\s*:\s*i64'
    pattern_exp = r'exp\s*=\s*([0-9]+)\s*:\s*i64'

    note = "\n// ======================================================================\n"
    note += "// Declaration of "
    note += funcName + "\n"
    note += "// ======================================================================\n"
    print(note)

    declare = "Value " \
            + "input0_" + idx + "," \
            + "input0_" + idx + "," \
            + "output0_" + idx + ";"
    # declare += "QScale scale_" + idx + ";\n"
    print(declare)

    # input0
    match = re.search(pattern_input0, line)
    define0,define1,defineout,define= "","","",""
    if match:
        shape = match.group(1)
        define0 = "std::vector<int> shape0_" + idx\
                + "= { " + shape + " };"
    else:
        assert 0
    # input1
    match = re.search(pattern_input1, line)
    if match:
        shape = match.group(1)
        define1 = "std::vector<int> shape1_" + idx\
                + "= { " + shape + " };"
    else:
        assert 0
    # output
    match = re.search(pattern_output, line)
    if match:
        shape = match.group(1)
        defineout = "std::vector<int> out_shape_" + idx\
                + "= { " + shape + " };"
    else:
        assert 0
    # exp
    match = re.search(pattern_exp, line)
    if match:
        exp = match.group(1)
        # definescale += "scale_" + idx + ".exp = " + exp + ";\n" 
    else:
        assert 0
    # m
    match = re.search(pattern_m, line)
    if match:
        m = match.group(1)
        # definescale += "scale_" + idx + ".m = " + m + ";\n" 
    else:
        assert 0
    definescale = "QScale scale_" + idx + " { " \
                + exp + "," + m + ", 8 };\n"
    # definescale += "scale_" + idx + ".qbits = 8;\n" 

    print(define0)
    print(define1)
    print(defineout)
    # print(define)
    print(definescale)



    define = "Matmul matmul_" + idx + ";\n"
    define += "Result result_" + idx + ";\n"
    print(define)

def codeGenCompileMatmul(idx_list):
    print(
        """
// ======================================================================
// Compile all matmul_accel
// ======================================================================

extern \"C\" void CompileMatmul(){
    HINSTANCE  handle = LoadLibraryA("./dll/icraft_buyi_taskgen.dll");
    if (!handle) {
        printf("Failed to load library icraft_buyi_taskgen %d\\n", GetLastError());
        return -1;
    }
    typedef void(*CompileMatMul)(Result*, Matmul*);       
    auto func = (CompileMatMul)GetProcAddress(handle, "CompileMatMul");
    if (!func) {
        printf("Failed to get function address\\n");
        return -1;
    } 
        """
    )
    for idx in idx_list:
        note = "\n    // ======================================================================\n"
        note += "    // Compilation of matmul_accel_" + idx + "\n"
        note += "    // ======================================================================\n"
        define = ""
        define += "    input0_" + idx \
                + ".shape = shape0_" + idx \
                + ".data();\n"
        define += "    input0_" + idx \
                + ".shape_size = shape0_" + idx \
                + ".size();\n"      
        define += "    input1_" + idx \
                + ".shape = shape1_" + idx \
                + ".data();\n"
        define += "    input1_" + idx \
                + ".shape_size = shape1_" + idx \
                + ".size();\n"
        define += "    output_" + idx \
                + ".shape = out_shape_" + idx \
                + ".data();\n"
        define += "    output_" + idx \
                + ".shape_size = out_shape_" + idx \
                + ".size();\n\n"
                
        define += "    Value inputs_arr_" + idx \
            + "[2] = { " \
            + "input0_" + idx \
            + ", input1_" + idx + " };\n"
        define += "    Value* inputs_ptr_" + idx \
            + " = inputs_arr_" + idx + ";\n\n"

        define += "    matmul_" + idx + ".inputs_size = 2;\n"
        define += "    matmul_" + idx + ".inputs = inputs_ptr_" + idx + ";\n"
        define += "    matmul_" + idx + ".output = output_" + idx + ";\n"
        define += "    matmul_" + idx + ".scale = scale_" + idx + ";\n"
        define += "    matmul_" + idx + ".byte_base = " + idx * 4096 +  " + 4096;\n" 
        func_exe =  "    func(&result_" + idx + ", &matmul_" + idx + ");\n"
        print(note)
        print(define)
        print(func_exe)

    print("""
    FreeLibrary(handle);
}
    """)

def codeGenMatmulAccelFunc(idx_list):
    for idx in idx_list:
        note = "\n// ======================================================================\n"
        note += "// Implementation of matmul_accel_" + idx + "\n"
        note += "// ======================================================================\n"
       
        func_head = "extern \"C\" void Matmul_accel_" + idx \
            + "(float* input0, float* input1, float* output){" \
            + \
            """
    HINSTANCE  handle_sim = LoadLibraryA("./dll/matmul_sim.dll");
    if (!handle_sim) {
        printf("Failed to load library handle_sim %d\\n", GetLastError());
        return -1;
    }
    typedef void(*MatMul)(SimValue*, SimValue*, SimValue*, CODE*);
    // get function address
    auto func_sim = (MatMul)GetProcAddress(handle_sim, "MatMul");
    if (!func_sim) {
        printf("Failed to get function address\\n");
        return -1;
    }\n""" \
            + """
    SimValue sim_input0, sim_input1, sim_output;\n""" \
            + """
    CODE sim_code;\n"""

        sim_input0= "    sim_input0.data = input0;\n" \
                +   "    sim_input0.byte_base = result_"+ idx +".input0_byte_addr;\n" \
                +   "    sim_input0.dim = shape0_" + idx + ".data();\n" \
                +   "    sim_input0.dim_size = input0_" + idx + ".shape_size;\n"

        sim_input1= "    sim_input1.data = input1;\n" \
                +   "    sim_input1.byte_base = result_"+ idx +".input1_byte_addr;\n" \
                +   "    sim_input1.dim = shape1_" + idx + ".data();\n" \
                +   "    sim_input1.dim_size = input1_" + idx + ".shape_size;\n"
        
        sim_output ="    sim_output.data = output;\n" \
                +   "    sim_output.byte_base = result_"+ idx +".output_byte_addr;\n" \
                +   "    sim_output.dim = out_shape_" + idx + ".data();\n" \
                +   "    sim_output.dim_size = output_" + idx + ".shape_size;\n"
        
        funcsim = "    func_sim(&sim_input0, &sim_input1, &sim_output, &sim_code);"
        print(note)
        print(func_head)
        print(sim_input0)
        print(sim_input1)
        print(sim_output)
        print(funcsim)
        print("\n}\n")


def codeGenMainFunc():
    note = "\n// ======================================================================\n"
    note += "// Main function\n"
    note += "// ======================================================================\n"
       
    func_head = "extern \"C\" int main(){\n" \
        + "    CompileMatmul();\n" \
        + "    // =========================================================\n" \
        + "    // Following code region can be defined by users.\n" \
        + "    // Don't Change any other auto-gen code region in this file.\n" \
        + "    // =========================================================\n"
    print(note)
    print(func_head)
    print("\n}\n")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('provide the source llvm-mlir file name')

    else:
        sourceFileName = sys.argv[1]
    
        with open(sourceFileName) as sourceFile:
            codeGenHeader()

            declares = []
            defines = []
            func_pattern = r'([a-zA-Z_]+)_(\d+)'
            idx_list = []

            lines = sourceFile.readlines()
            for line in lines:

                if "func.func " in line:
                    funcName = line.split("@")[1].split("(")[0]
                    match = re.match(func_pattern, funcName)

                    if match:
                        # print(funcName)
                        matmul_accel = match.group(1) 
                        idx = match.group(2) 
                        codeGenGlobalDeclares(line)
                        idx_list.append(idx)
            codeGenCompileMatmul(idx_list)
            codeGenMatmulAccelFunc(idx_list)
            codeGenMainFunc()

  
