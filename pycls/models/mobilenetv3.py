"""MobileNetV3 models."""

# import torch.nn as nn
# from pycls.core.config import cfg
# from pycls.models.blocks import (
#     activation,
#     conv2d,
#     conv2d_cx,
#     gap1d,
#     gap2d,
#     gap2d_cx,
#     init_weights,
#     linear,
#     linear_cx,
#     norm2d,
#     norm2d_cx,
#     make_divisible,
# )
# from torch.nn import Dropout, Module
# from torch.nn.quantized import FloatFunctional
# from torch.quantization import fuse_modules
# from torch.nn import functional as F


# class h_sigmoid(Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)

#     def forward(self, x):
#         return self.relu(x+3)/6


# class h_swish(Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)

#     def forward(self, x):
#         return x * self.sigmoid(inplace=inplace)


# class SELayer(Module):
#     """MobileNetV3 inverted sqeeze-and-excite"""

#     def __init__(self, channel, reduction=4):
#         # super().__init__()
#         # self.avg_pool = gap2d(1)
#         # self.fc1 = linear(channel, make_divisible(
#         #     channel//reduction, 8), bias=True)
#         # self.af1 = nn.ReLU(inplace=True)
#         # self.fc2 = linear(make_divisible(
#         #     channel//reduction, 8), channel, bias=True)
#         # self.af2 = F.hardsigmoid(inplace=inplace)
#         super().__init__()
#         self.avg_pool = gap2d(1)
#         self.fc1 = nn.Conv2d(channel, make_divisible(channel//reduction, 8), 1)
#         self.af1 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(make_divisible(channel//reduction, 8), channel, 1)
#         self.af2 = h_sigmoid()

#     def forward(self, x):
#         # b, c, _, _ = x.size()
#         # se = self.avg_pool(x).view(b, c)
#         # se = self.af2(self.fc2(self.af1(self.fc1(se)))).view(b, c, 1, 1)
#         se = self.af2(self.fc2(self.af1(self.fc1(self.avg_pool(x)))))
#         return x * se

#     @staticmethod
#     def complexity(cx, channel, reduction=4):
#         cx = gap2d_cx(cx, 1)
#         cx = conv2d_cx(cx, channel, make_divisible(channel//reduction, 8), 1)
#         cx = conv2d_cx(cx, make_divisible(channel//reduction, 8), channel, 1)
#         return cx


# class StemImageNet(Module):
#     """MobileNetV3 stem for ImageNet: 3x3, BN, AF(Hswish)."""

#     def __init__(self, w_in, w_out):
#         super(StemImageNet, self).__init__()
#         self.conv = conv2d(w_in, w_out, 3, stride=2)
#         self.bn = norm2d(w_out)
#         self.af = h_swish()

#     def forward(self, x):
#         x = self.af(self.bn(self.conv(x)))
#         return x

#     @staticmethod
#     def complexity(cx, w_in, w_out):
#         cx = conv2d_cx(cx, w_in, w_out, 3, stride=2)
#         cx = norm2d_cx(cx, w_out)
#         return cx

#     def fuse_model(self, include_relu: bool):
#         targets = [["conv", "bn", "af"]] if include_relu else [["conv", "bn"]]
#         fuse_modules(self, targets, inplace=True)


# class MBConv(Module):
#     """MobileNetV3 inverted bottleneck block"""

#     def __init__(self, w_in, exp_s, stride, w_out, nl, se, ks):
#         # Expansion, 3x3 depthwise, BN, AF, 1x1 pointwise, BN, skip_connection
#         super(MBConv, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]  # stride must be 1 or 2
#         self.exp = None
#         self.se = se
#         self.exp_s = exp_s
#         self.ks = ks
#         # expand
#         if exp_s != w_in:  # skip if exp_r is 1
#             self.exp = conv2d(w_in, exp_s, 1)
#             self.exp_bn = norm2d(exp_s)
#             self.exp_af = h_swish() if nl == 1 else nn.ReLU()
#         # depthwise
#         self.dwise = conv2d(exp_s, exp_s, k=ks,
#                             stride=stride, groups=exp_s)
#         self.dwise_bn = norm2d(exp_s)
#         self.dwise_af = h_swish() if nl == 1 else nn.ReLU()
#         # squeeze-and-excite
#         if self.se == 1:
#             self.selayer = SELayer(exp_s)
#         # pointwise
#         self.lin_proj = conv2d(exp_s, w_out, 1)
#         self.lin_proj_bn = norm2d(w_out)

#         self.use_res_connect = (
#             self.stride == 1 and w_in == w_out
#         )  # check skip connection
#         if self.use_res_connect:
#             self.skip_add = FloatFunctional()

#     def forward(self, x):
#         f_x = self.exp_af(self.exp_bn(self.exp(x))) if self.exp else x
#         f_x = self.dwise_af(self.dwise_bn(self.dwise(f_x)))
#         f_x = self.selayer(f_x) if self.se == 1 else f_x
#         f_x = self.lin_proj_bn(self.lin_proj(f_x))
#         if self.use_res_connect:
#             f_x = self.skip_add.add(x, f_x)
#         return f_x

#     @staticmethod
#     def complexity(cx, w_in, exp_s, stride, w_out, se, ks):
#         # w_exp = int(w_in * exp_r)  # expand channel using expansion factor
#         if exp_s != w_in:
#             cx = conv2d_cx(cx, w_in, exp_s, 1)
#             cx = norm2d_cx(cx, exp_s)
#         # depthwise
#         cx = conv2d_cx(cx, exp_s, exp_s, k=ks, stride=stride, groups=exp_s)
#         cx = norm2d_cx(cx, exp_s)
#         # squeeze-and-excite
#         cx = SELayer.complexity(cx, exp_s) if se == 1 else cx
#         # pointwise
#         cx = conv2d_cx(cx, exp_s, w_out, 1)
#         cx = norm2d_cx(cx, w_out)
#         return cx

#     def fuse_model(self, include_relu: bool):
#         targets = (
#             [["dwise", "dwise_bn", "dwise_af"], ["lin_proj", "lin_proj_bn"]]
#             if include_relu
#             else [["dwise", "dwise_bn"], ["lin_proj", "lin_proj_bn"]]
#         )
#         if self.exp:
#             targets.append(
#                 ["exp", "exp_bn", "exp_af"] if include_relu else ["exp", "exp_bn"]
#             )
#         fuse_modules(self, targets, inplace=True)


# class MNV3Stage(Module):
#     """MobileNetV3 stage."""

#     def __init__(self, w_in, exp_s, stride, w_out, nl, se, ks):
#         super(MNV3Stage, self).__init__()
#         stride = stride
#         block = MBConv(w_in, exp_s, stride, w_out, nl, se, ks)
#         self.add_module("b{}".format(1), block)
#         stride, w_in = 1, w_out

#     def forward(self, x):
#         for block in self.children():
#             x = block(x)
#         return x

#     @staticmethod
#     def complexity(cx, w_in, exp_r, stride, w_out, se, ks):
#         stride = stride
#         cx = MBConv.complexity(cx, w_in, exp_r, stride, w_out, se, ks)
#         stride, w_in = 1, w_out
#         return cx

#     def fuse_model(self, include_relu: bool):
#         for m in self.modules():
#             if type(m) == MBConv:
#                 m.fuse_model(include_relu)


# class MNV3Head(Module):
#     """MobileNetV3 head: 1x1, BN, AF(ReLU6), AvgPool, FC, Dropout, FC."""

#     def __init__(self, w_in, w_out, num_classes, exp_s):
#         super(MNV3Head, self).__init__()
#         dropout_ratio = cfg.MNV2.DROPOUT_RATIO
#         self.conv = conv2d(w_in, exp_s, 1)
#         self.conv_bn = norm2d(exp_s)
#         self.conv_af = h_swish()
#         self.avg_pool = gap2d(exp_s)
#         # classifier
#         self.fc1 = linear(exp_s, w_out, bias=True)
#         self.cf_af = h_swish()
#         self.dropout = Dropout(p=dropout_ratio) if dropout_ratio > 0 else None
#         self.fc2 = linear(w_out, num_classes, bias=True)

#     def forward(self, x):
#         x = self.conv_af(self.conv_bn(self.conv(x)))
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.cf_af(self.fc1(x))
#         x = self.dropout(x) if self.dropout else x
#         x = self.fc2(x)
#         return x

#     @staticmethod
#     def complexity(cx, w_in, w_out, num_classes, exp_s):
#         cx = conv2d_cx(cx, w_in, w_out, 1)
#         cx = norm2d_cx(cx, w_out)
#         cx = gap2d_cx(cx, w_out)
#         cx = linear_cx(cx, exp_s, w_out, bias=True)
#         cx = linear_cx(cx, w_out, num_classes, bias=True)
#         return cx

#     def fuse_model(self, include_relu: bool):
#         targets = (
#             [["conv", "conv_bn", "conv_af"]] if include_relu else [
#                 ["conv", "conv_bn"]]
#         )
#         fuse_modules(self, targets, inplace=True)


# class MobileNetV3(Module):
#     """"MobileNetV3 model."""

#     @staticmethod
#     def get_params():
#         return {
#             "sw": cfg.MNV2.STEM_W,
#             "ws": cfg.MNV2.WIDTHS,
#             "ss": cfg.MNV2.STRIDES,
#             "hw": cfg.MNV2.HEAD_W,
#             "nc": cfg.MODEL.NUM_CLASSES,
#         }

#     def __init__(self, params=None):
#         super(MobileNetV3, self).__init__()
#         p = MobileNetV3.get_params() if not params else params

#         # parameters of MobileNetV3_large
#         # MNV2.STEM_W 16 \
#         # MNV2.STRIDES '[1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1]' \
#         # MNV2.WIDTHS '[16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160]' \
#         # MNV2.HEAD_W 1280 \
#         # MNV2.NUM_CLASSES 1000

#         p["exp_sz"] = [16, 64, 72, 72, 120, 120, 240,
#                        200, 184, 184, 480, 672, 672, 960, 960]
#         p["nl"] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#         p["se"] = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
#         p["ks"] = [3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5]
#         vs = ["sw", "ws", "ss", "hw", "nc", "exp_sz", "nl", "se", "ks"]
#         sw, ws, ss, hw, nc, exp_sz, nl, se, ks = [p[v] for v in vs]
#         stage_params = list(zip(ws, ss, exp_sz, nl, se, ks))
#         self.stem = StemImageNet(3, sw)
#         prev_w = sw
#         for i, (w, stride, exp_s, nl, se, ks) in enumerate(stage_params):
#             stage = MNV3Stage(prev_w, exp_s, stride, w, nl, se, ks)
#             self.add_module("s{}".format(i + 1), stage)
#             prev_w = w
#         self.head = MNV3Head(prev_w, hw, nc, exp_s)
#         self.apply(init_weights)

#     def forward(self, x):
#         for module in self.children():
#             x = module(x)
#         return x

#     @staticmethod
#     def complexity(cx, params=None):
#         """Computes model complexity (if you alter the model, make sure to update)."""
#         p = MobileNetV3.get_params() if not params else params
#         p["exp_sz"] = [16, 64, 72, 72, 120, 120, 240,
#                        200, 184, 184, 480, 672, 672, 960, 960]
#         p["nl"] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#         p["se"] = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
#         p["ks"] = [3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5]
#         vs = ["sw", "ws", "ss", "hw", "nc", "exp_sz", "nl", "se", "ks"]
#         sw, ws, ss, hw, nc, exp_sz, nl, se, ks = [p[v] for v in vs]
#         stage_params = list(zip(ws, ss, exp_sz, nl, se, ks))
#         cx = StemImageNet.complexity(cx, 3, sw)
#         prev_w = sw
#         for w, stride, exp_s, nl, se, ks in stage_params:
#             cx = MNV3Stage.complexity(cx, prev_w, exp_s, stride, w, se, ks)
#             prev_w = w
#         cx = MNV3Head.complexity(cx, prev_w, hw, nc, exp_s)
#         return cx

#     def fuse_model(self, include_relu: bool = False):
#         self.stem.fuse_model(include_relu)
#         for m in self.modules():
#             if type(m) == MNV3Stage:
#                 m.fuse_model(include_relu)
#         self.head.fuse_model(include_relu)

#     def postprocess_skip(self):
#         for mod in self.modules():
#             if isinstance(mod, MNV3Stage) and len(mod._modules) > 1:
#                 fakeq = mod._modules["b1"].lin_proj.activation_post_process
#                 for _, m in list(mod._modules.items())[1:]:
#                     m.lin_proj.activation_post_process = fakeq

import torch

from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence

from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.mobilenetv2 import _make_divisible, ConvBNActivation


__all__ = ["MobileNetV3", "mobilenet_v3_large", "mobilenet_v3_small"]


model_urls = {
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}


class SqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
                 activation: str, stride: int, dilation: int, width_mult: float):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(
            expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                       stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                       norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels))

        # project
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(Module):
    def __init__(
        self,
        reduece_divider=1,
        dilation=1,
        bneck_conf=partial(InvertedResidualConfig, width_mult=1.0),
        adjust_channels=partial(
            InvertedResidualConfig.adjust_channels, width_mult=1.0),
        inverted_residual_setting: List[InvertedResidualConfig] = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider,
                       True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider,
                       160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider,
                       160 // reduce_divider, True, "HS", 1, dilation),
        ],
        last_channel: int = adjust_channels(1280 // reduce_divider),  # C5,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError(
                "The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError(
                "The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
