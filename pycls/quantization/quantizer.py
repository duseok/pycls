from copy import deepcopy

import torch
import torch.nn as nn


class QuantizedModel(nn.Module):
    def __init__(self, model_fp32: nn.Module):
        super(QuantizedModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = deepcopy(model_fp32)

    def postprocess_skip(self):
        self.model_fp32.postprocess_skip()

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x
