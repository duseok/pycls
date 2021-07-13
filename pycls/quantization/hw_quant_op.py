from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycls.core.config import cfg
from pycls.quantization.quant_op import QConv2d, QConvBn2d, QLinear
from torch.nn.modules.pooling import AdaptiveAvgPool2d, MaxPool2d
from torch.nn.quantized.modules.functional_modules import FloatFunctional


class Quantizer(nn.Module):
    def forward(self, input, bq, s):
        x = input.div(s).round_()
        if bq is not None:
            _bq = bq.div(s).round_()
            x = x + _bq
        out = x.clamp(
            min=-int(np.exp2(cfg.QUANTIZATION.QAT.ACT_BITWIDTH - 1)),
            max=int(np.exp2(cfg.QUANTIZATION.QAT.ACT_BITWIDTH - 1) - 1),
        ).mul_(s)
        return out


class HWQConv2d(nn.Conv2d):
    def __init__(self, *args, **kargs):
        super(HWQConv2d, self).__init__(*args, **kargs)
        self.quant = Quantizer()
        self.scale = 1.0

    def initialize(self, s):
        self.scale = s

    def forward(self, x):
        y = F.conv2d(
            x, self.weight, None, self.stride, self.padding, self.dilation, self.groups
        )
        return self.quant(y, self.bias, self.scale)

    @classmethod
    def from_trained_op(cls, op):
        midap_op = cls(
            op.in_channels,
            op.out_channels,
            op.kernel_size,
            stride=op.stride,
            padding=op.padding,
            dilation=op.dilation,
            groups=op.groups,
            bias=(op.bias is not None),
            padding_mode=op.padding_mode,
        )
        act_scale = _get_shift_scale_value(op.activation_post_process.scale)
        w_scale = _get_shift_scale_value(op.weight_fake_quant.scale)
        midap_op.initialize(act_scale)
        midap_op.weight.data = torch.fake_quantize_per_tensor_affine(
            op.weight,
            float(w_scale),
            0,
            -int(np.exp2(cfg.QUANTIZATION.QAT.WEIGHT_BITWIDTH - 1)),
            int(np.exp2(cfg.QUANTIZATION.QAT.WEIGHT_BITWIDTH - 1) - 1),
        )
        midap_op.bias.data = torch.fake_quantize_per_tensor_affine(
            op.bias.reshape(op.out_channels, 1, 1),
            float(act_scale),
            0,
            -int(np.exp2(cfg.QUANTIZATION.QAT.ACT_BITWIDTH - 1)),
            int(np.exp2(cfg.QUANTIZATION.QAT.ACT_BITWIDTH - 1) - 1),
        )
        return midap_op


class HWQConvBn2d(HWQConv2d):
    @classmethod
    def from_trained_op(cls, op):
        from torch.quantization import fuse_conv_bn

        op.eval()
        conv = fuse_conv_bn(op, op.bn)
        return HWQConv2d.from_trained_op(conv)


class HWQLinear(nn.Linear):
    def __init__(self, *args, **kargs):
        super(HWQLinear, self).__init__(*args, **kargs)
        self.quant = Quantizer()
        self.scale = 1.0

    def initialize(self, s):
        self.scale = s

    def forward(self, x):
        y = F.linear(x, self.weight, None)
        return self.quant(y, self.bias, self.scale)

    @classmethod
    def from_trained_op(cls, op):
        midap_op = cls(op.in_features, op.out_features)
        act_scale = _get_shift_scale_value(op.activation_post_process.scale)
        w_scale = _get_shift_scale_value(op.weight_fake_quant.scale)
        midap_op.initialize(act_scale)
        midap_op.weight.data = torch.fake_quantize_per_tensor_affine(
            op.weight,
            float(w_scale),
            0,
            -int(np.exp2(cfg.QUANTIZATION.QAT.WEIGHT_BITWIDTH - 1)),
            int(np.exp2(cfg.QUANTIZATION.QAT.WEIGHT_BITWIDTH - 1) - 1),
        )
        midap_op.bias.data = torch.fake_quantize_per_tensor_affine(
            op.bias,
            float(act_scale),
            0,
            -int(np.exp2(cfg.QUANTIZATION.QAT.ACT_BITWIDTH - 1)),
            int(np.exp2(cfg.QUANTIZATION.QAT.ACT_BITWIDTH - 1) - 1),
        )
        return midap_op


class HWQFunctional(nn.Module):
    def __init__(self, *args, **kargs):
        super(HWQFunctional, self).__init__(*args, **kargs)
        self.quant = Quantizer()
        self.scale = 1.0

    def initialize(self, s):
        self.scale = s

    def add(self, x1, x2):
        y = x1 + x2
        return self.quant(y, None, self.scale)

    def mul(self, x1, x2):
        y = x1 * x2
        return self.quant(y, None, self.scale)

    @classmethod
    def from_trained_op(cls, op):
        midap_op = cls()
        scale = _get_shift_scale_value(op.activation_post_process.scale)
        midap_op.initialize(scale)
        return midap_op


class HWQAvgPool2d(nn.Module):
    def __init__(self, *args, **kargs):
        super(HWQAvgPool2d, self).__init__(*args, **kargs)
        self.divisor_override = 1

    def forward(self, x):
        y = torch.sum(x, (2, 3), keepdim=True)
        size = np.exp2(np.ceil(np.log2(x.shape[-1] * x.shape[-2])))
        return y.div_(size)

    @classmethod
    def from_trained_op(cls, op=None):
        return cls()


class HWQMaxPool2d(nn.MaxPool2d):
    def __init__(self, *args, **kargs):
        super(HWQMaxPool2d, self).__init__(*args, **kargs)

    @classmethod
    def from_trained_op(cls, op):
        return cls(op.kernel_size, op.stride, op.padding)


def _get_shift_scale_value(s):
    return torch.exp2(torch.log2(F.softplus(s)).round_()).cuda()


QuantOps = {
    QConv2d: HWQConv2d,
    QConvBn2d: HWQConvBn2d,
    MaxPool2d: HWQMaxPool2d,
    AdaptiveAvgPool2d: HWQAvgPool2d,
    FloatFunctional: HWQFunctional,
    QLinear: HWQLinear,
}
