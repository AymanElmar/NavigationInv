import copy
import numpy as np

import torch
import torch.nn as nn



def init(module: nn.Module, weight_init, bias_init, gain: float = 1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module: nn.Module, N: int):  
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])