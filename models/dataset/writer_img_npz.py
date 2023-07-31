# !/usr/bin/env python
import numpy as np


if __name__ == '__main__':
    array = np.ones((1, 3, 16, 16))
    tensor_dic={}
    tensor_dic["input"]=array
    tensor_file = "1x3x16x16.npz"
    np.savez(tensor_file, **tensor_dic)