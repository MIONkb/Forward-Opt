# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import subprocess
import logging


def _os_system_log(cmd_str):
    # use subprocess to redirect the output stream
    # the file for saving the output stream should be set if using this function
    logging.info("[Running]: %s", cmd_str)

    process = subprocess.Popen(cmd_str,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)

    while True:
        output = process.stdout.readline().strip()
        if output == '' and process.poll() is not None:
            break
        if output:
            logging.info(output)

    process.wait()
    ret = process.returncode

    if ret == 0:
        logging.info("[Success]: %s", cmd_str)
    else:
        raise RuntimeError("[!Error]: {}".format(cmd_str))


def _os_system(cmd: list, save_log: bool = False):
    cmd_str = ""
    for s in cmd:
        cmd_str += str(s) + " "
    if not save_log:
        print("[Running]: {}".format(cmd_str))
        ret = os.system(cmd_str)
        if ret == 0:
            print("[Success]: {}".format(cmd_str))
        else:
            raise RuntimeError("[!Error]: {}".format(cmd_str))
    else:
        _os_system_log(cmd_str)


def mlir_opt_for_top(mlirfile, opt_mlirfile, add_postprocess=""):
    cmd = ["tpuc-opt", mlirfile, "--shape-infer"]
    if len(add_postprocess) > 0:
        cmd.extend([f"--add-postprocess=\"type={add_postprocess}\""])
    cmd.extend(["--canonicalize", "--extra-optimize", "-o", opt_mlirfile])
    _os_system(cmd)


def mlir_lowering(top_mlir: str,
                  tpu_mlir: str,
                  mode: str,
                  chip: str,
                  cali_table: str = None,
                  asymmetric: bool = False,
                  quantize_table: str = None,
                  customization_format: str = None,
                  fuse_preprocess: bool = False,
                  aligned_input: bool = False):
    cmd = ["tpuc-opt", top_mlir, "--chip-assign=\"chip={}\"".format(chip.lower())]
    mode = mode.upper()
    asymmetric = False # TODO: always using symmetric, as asymmetric not good
    if cali_table != None:
        cali_param = "--import-calibration-table=\"file={} asymmetric={}\"".format(
            cali_table, asymmetric)
        cmd.extend([cali_param])
    #do extra conversion for differnet chips
    cmd.extend(["--chip-top-optimize"])
    if fuse_preprocess:
        fuse_pre_param = "--fuse-preprocess=\"mode={} customization_format={} align={}\"".format(
            mode, customization_format, aligned_input)
        cmd.extend([fuse_pre_param])
    qtable = ""
    if quantize_table:
        assert (tpu_mlir.endswith(".mlir"))
        weight_name = tpu_mlir[:-len(".mlir")] + "_qtable_weights.npz"
        qtable = "qtable={} weightFileName={}".format(quantize_table, weight_name)
    lower_param = "--convert-top-to-tpu=\"mode={} {} asymmetric={}\"".format(
        mode, qtable, asymmetric)
    cmd.extend([
        lower_param,
        "--canonicalize",
        "-o",
        tpu_mlir,
    ])
    _os_system(cmd)


def mlir_to_model(tpu_mlir: str,
                  model: str,
                  final_mlir: str,
                  dynamic: bool = False,
                  quant_input: bool = False,
                  quant_output: bool = False,
                  disable_layer_group: bool = False,
                  merge_weight: bool = False,
                  op_divide: bool = False):
    # generate final mlir
    strip_io_quant_param = '--strip-io-quant="quant_input={} quant_output={}"'.format(
        quant_input, quant_output)
    lg_param = ''
    if not disable_layer_group:
        lg_param = '--layer-group="opt=2"'
    subnet_param = '--subnet-divide="dynamic={}"'.format(dynamic)
    address_assign_param = '--address-assign'
    #address_assign_param = '--address-assign="reuse_addr=false"'
    if merge_weight:
        address_assign_param = '--address-assign="merge_weight=true weight_map_file=_weight_map.csv"'
    op_divide_param = ""
    if op_divide:
        op_divide_param = "--op-divide"
    cmd = [
        "tpuc-opt",
        tpu_mlir,
        "--mlir-disable-threading",
        strip_io_quant_param,
        "--chip-tpu-optimize",
        "--weight-reorder",
        op_divide_param,
        subnet_param,
        "--op-reorder",
        lg_param,
        address_assign_param,
        "-o",
        final_mlir,
    ]

    _os_system(cmd)

    # codegen based on final mlir
    codegen_param = '--codegen="model_file={}"'.format(model)
    cmd = [
        "tpuc-opt",
        final_mlir,
        codegen_param,
        "-o /dev/null",
    ]
    _os_system(cmd)

    try:
        if model.endswith(".bmodel"):
            # The suffix of the profile file is not consistent.
            # bm1684 uses ".dat", bm1684x uses ".txt".
            _os_system(["mv compiler_profile_0.[td][xa]t", model + ".compiler_profile_0.txt"])
            _os_system(["mv net_0.profile", model + ".net_0.profile"])
    except RuntimeError:
        pass


def f32_blobs_compare(a_npz: str, b_npz: str, tolerance: str, excepts=None, show_detail=True):
    cmd = ["npz_tool.py", "compare", a_npz, b_npz, "--tolerance", tolerance]
    if excepts:
        cmd.extend(["--except", excepts])
    if show_detail:
        cmd.append('-vv')
    _os_system(cmd)


# TOPTOTOSA
def top2tosa(
    top_mlir: str,
    tosa_mlir: str,
):
    cmd = ["tpuc-opt", top_mlir]
    lower_param = "--convert-top-to-tosa"
    cmd.extend([
        lower_param,
        "--canonicalize",
        "-o",
        tosa_mlir,
    ])
    _os_system(cmd)
