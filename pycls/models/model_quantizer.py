from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.quantization.observer import MinMaxObserver


class QuantizedModel(nn.Module):
    def __init__(self, model_fp32: nn.Module):
        super(QuantizedModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = deepcopy(model_fp32)

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x


class MinMaxShiftObserver(MinMaxObserver):
    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch._aminmax(x)

        max_abs = max(abs(min_val_cur), abs(max_val_cur))
        value = int(np.floor(np.log2(max_abs / 255)))
        delta = abs(max_abs - 255 * (2 ** value))
        while 255 * (2 ** value) < max_abs:
            value += 1
            delta = abs(max_abs - 255 * (2 ** value))
        if delta > abs(max_abs - 255 * (2 ** (value + 1))):
            value += 1

        min_val = torch.min(torch.tensor(-(255 * (2 ** value))), self.min_val)
        max_val = torch.max(torch.tensor(255 * (2 ** value)), self.max_val)

        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig
