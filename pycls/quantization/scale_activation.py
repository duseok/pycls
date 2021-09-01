from abc import abstractstaticmethod

import torch
import torch.nn.functional as F
from pycls.core.config import cfg
from torch import Tensor


class ScaleAcitvation:
    @abstractstaticmethod
    def apply(x: Tensor):
        pass

    @abstractstaticmethod
    def inverse(x: Tensor):
        pass


class Softplus(ScaleAcitvation):
    def apply(x: Tensor):
        return F.softplus(x)

    def inverse(x: Tensor):
        return torch.log(torch.exp(x) - 1)


class Sigmoid(ScaleAcitvation):
    def apply(x: Tensor):
        return x.sigmoid()

    def inverse(x: Tensor):
        return torch.log(x / (1 - x))


def get_scale_act():
    if cfg.QUANTIZATION.QAT.SCALE_ACT == "softplus":
        return Softplus
    elif cfg.QUANTIZATION.QAT.SCALE_ACT == "sigmoid":
        return Sigmoid
    raise ValueError(f"Only support softplus and sigmoid for the scale activation function")
