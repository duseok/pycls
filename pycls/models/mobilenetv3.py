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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SELayer(Module):
    """MobileNetV3 inverted sqeeze-and-excite"""

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV3(Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )


        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out

