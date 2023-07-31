# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torch import nn
import torchvision
from torchvision import ops

import torch_mlir

class simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torchvision.ops.Conv2dNormActivation(in_channels = 3, out_channels = 8, kernel_size = 3, 
                                                         bias = True)
        self.mlp = torchvision.ops.MLP(in_channels = 16, hidden_channels=[8,4],bias = True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.mlp(x)
        return x

ins = simple()
ins.eval()
print(ins)

r = ins(torch.ones(1, 3, 16, 16))

# 打开文件以供写入
file = open("TORCH_test.mlir", "w")
# # 将输出打印到文件
module = torch_mlir.compile(ins, torch.ones(1, 3, 16, 16), output_type="torch", use_tracing = True)
print(module.operation.get_asm(large_elements_limit=2), file=file)
# # 关闭文件
file.close()


# 打开文件以供写入
file = open("LINALG_TENSOR.mlir", "w")
module = torch_mlir.compile(ins, torch.ones(1, 3, 16, 16), output_type="linalg-on-tensors", use_tracing = True)
# print(module.operation.get_asm(large_elements_limit=10), file=file)
print(module.operation.get_asm(), file=file)
# # 关闭文件
file.close()

# # 打开文件以供写入
# file = open("TOSA_test.mlir", "w")
# module = torch_mlir.compile(ins, torch.ones(1, 3, 16, 16), output_type="tosa", use_tracing = True)
# print(module.operation.get_asm(large_elements_limit=2), file=file)
# # # 关闭文件
# file.close()

# module = torch_mlir.compile(resnet18, torch.ones(1, 3, 224, 224), output_type="RAW")
# print("LINALG_ON_TENSORS OutputType\n", module.operation.get_asm(large_elements_limit=10))
# TODO: Debug why this is so slow.

