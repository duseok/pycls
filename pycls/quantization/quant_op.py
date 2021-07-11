from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.qat import Conv2d, Linear


class ShiftScaleQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s):
        return torch.exp2(torch.log2(s).round())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, bias, scale) -> Any:
        qbias = bias.div(scale).round_()
        qbias = qbias.clamp(min=-128, max=127).mul_(scale)
        return qbias

    @staticmethod
    def backward(ctx: Any, grad_outputs) -> Any:
        return grad_outputs, None


class QConv2d(Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )
        self.quant_bias = False

    def set_quant_bias(self, quant_bias):
        self.quant_bias = quant_bias

    def forward(self, input):
        qweight = self.weight_fake_quant(self.weight)
        qbias = self.bias
        if self.bias is not None and self.quant_bias:
            scale = ShiftScaleQuant.apply(
                F.softplus(self.activation_post_process.scale)
            )
            qbias = RoundQuant.apply(self.bias, scale)
        return self._conv_forward(input, qweight, qbias)


class QLinear(Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )
        self.quant_bias = False

    def set_quant_bias(self, quant_bias):
        self.quant_bias = quant_bias

    def forward(self, input):
        qweight = self.weight_fake_quant(self.weight)
        qbias = self.bias
        if self.bias is not None and self.quant_bias:
            scale = ShiftScaleQuant.apply(
                F.softplus(self.activation_post_process.scale)
            )
            qbias = RoundQuant.apply(self.bias, scale)
        return F.linear(input, qweight, qbias)
