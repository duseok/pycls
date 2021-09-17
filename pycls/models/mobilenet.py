"""MobileNet models."""

from typing import KeysView
from torch.nn.modules.activation import Hardswish
from pycls.core.config import cfg
from pycls.models.blocks import (
    activation,
    conv2d,
    conv2d_cx,
    gap2d,
    gap2d_cx,
    init_weights,
    linear,
    linear_cx,
    norm2d,
    norm2d_cx,
    make_divisible,
)
from torch.nn import Dropout, Module, Hardsigmoid, Hardswish
from torch.nn.quantized import FloatFunctional
from torch.quantization import fuse_modules
from torch.nn import functional as F


class SELayer(Module):
    """MobileNetV3 inverted sqeeze-and-excite"""

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = gap2d(1)
        self.fc1 = conv2d(channel, make_divisible(channel//reduction, 8), 1)
        self.af1 = activation()
        self.fc2 = conv2d(make_divisible(channel//reduction, 8), channel, 1)
        self.af2 = Hardsigmoid()
        self.se_mul = FloatFunctional()

    def forward(self, x):
        se = self.af2(self.fc2(self.af1(self.fc1(self.avg_pool(x)))))
        return self.se_mul.mul(se, x)

    @staticmethod
    def complexity(cx, channel, reduction=4):
        cx = gap2d_cx(cx, 1)
        cx = conv2d_cx(cx, channel, make_divisible(channel//reduction, 8), 1)
        cx = conv2d_cx(cx, make_divisible(channel//reduction, 8), channel, 1)
        return cx

    def fuse_model(self, include_relu: bool):
        targets = [["fc1", "af1"]]
        fuse_modules(self, targets, inplace=True)


class StemImageNet(Module):
    """MobileNet stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out, nl):
        super(StemImageNet, self).__init__()
        self.nl = nl
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(w_out)
        self.af = Hardswish() if self.nl else activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=2)
        cx = norm2d_cx(cx, w_out)
        return cx

    def fuse_model(self, include_relu: bool):
        targets = [["conv", "bn", "af"]] if include_relu and not(self.nl) else [["conv", "bn"]]
        fuse_modules(self, targets, inplace=True)


class MBConv(Module):
    """MobileNet inverted bottleneck block"""

    def __init__(self, w_in, exp_s, stride, w_out, ks, nl, se):
        # Expansion, 3x3 depthwise, BN, AF, 1x1 pointwise, BN, skip_connection
        super(MBConv, self).__init__()
        self.stride = stride
        assert stride in [1, 2]  # stride must be 1 or 2
        self.exp = None
        self.se = se
        self.nl = nl
        # expand channel
        if exp_s != w_in: 
            self.exp = conv2d(w_in, exp_s, 1)
            self.exp_bn = norm2d(exp_s)
            self.exp_af = Hardswish() if self.nl else activation()
        # depthwise
        self.dwise = conv2d(exp_s, exp_s, k=ks, stride=stride, groups=exp_s)
        self.dwise_bn = norm2d(exp_s)
        self.dwise_af = Hardswish() if self.nl else activation()
        # squeeze-and-excite
        if self.se:
            self.selayer = SELayer(exp_s)
        # pointwise
        self.lin_proj = conv2d(exp_s, w_out, 1)
        self.lin_proj_bn = norm2d(w_out)
        self.use_res_connect = (
            self.stride == 1 and w_in == w_out
        )  # check skip connection
        if self.use_res_connect:
            self.skip_add = FloatFunctional()

    def forward(self, x):
        f_x = self.exp_af(self.exp_bn(self.exp(x))) if self.exp else x
        f_x = self.dwise_af(self.dwise_bn(self.dwise(f_x)))
        f_x = self.selayer(f_x) if self.se else f_x
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.use_res_connect:
            f_x = self.skip_add.add(x, f_x)
        return f_x

    @staticmethod
    def complexity(cx, w_in, exp_s, stride, w_out, ks, se):
        if exp_s != w_in:
            cx = conv2d_cx(cx, w_in, exp_s, 1)
            cx = norm2d_cx(cx, exp_s)
        # depthwise
        cx = conv2d_cx(cx, exp_s, exp_s, k=ks, stride=stride, groups=exp_s)
        cx = norm2d_cx(cx, exp_s)
        # squeeze-and-excite
        cx = SELayer.complexity(cx, exp_s) if se else cx
        # pointwise
        cx = conv2d_cx(cx, exp_s, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx

    def fuse_model(self, include_relu: bool):
        targets = (
            [["dwise", "dwise_bn", "dwise_af"], ["lin_proj", "lin_proj_bn"]]
            if (include_relu and not(self.nl))
            else [["dwise", "dwise_bn"], ["lin_proj", "lin_proj_bn"]]
        )
        if self.exp:
            targets.append(
                ["exp", "exp_bn", "exp_af"] if (include_relu and not(self.nl)) else ["exp", "exp_bn"]
            )
        if self.se:
            self.selayer.fuse_model(include_relu)
        fuse_modules(self, targets, inplace=True)


class MNStage(Module):
    """MobileNet stage."""

    def __init__(self, w_in, exp_s, stride, w_out, d, ks, nl, se):
        super(MNStage, self).__init__()

        for i in range(d):  # d는 layer의 반복 횟수
            stride = stride if i == 0 else 1
            block = MBConv(w_in, exp_s, stride, w_out, ks, nl, se)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, exp_s, stride, w_out, d, ks, se):
        for i in range(d):
            stride = stride if i == 0 else 1
            cx = MBConv.complexity(cx, w_in, exp_s, stride, w_out, ks, se)
            stride, w_in = 1, w_out
        return cx

    def fuse_model(self, include_relu: bool):
        for m in self.modules():
            if type(m) == MBConv:
                m.fuse_model(include_relu)


class MNHead(Module):
    """MobileNet head: 1x1, BN, AF(ReLU6), AvgPool, (mobilenetV3-additional FC), Dropout, FC."""

    def __init__(self, w_in, w_out, num_classes, exp_s, nl):
        super(MNHead, self).__init__()
        dropout_ratio = cfg.MN.DROPOUT_RATIO
        self.nl = nl
        exp_s = w_out if nl == 0 else exp_s
        self.conv = conv2d(w_in, exp_s, 1)
        self.conv_bn = norm2d(exp_s)
        self.conv_af = Hardswish() if nl else activation()
        self.avg_pool = gap2d(exp_s)
        # classifier
        if self.nl:
            self.fc1 = linear(exp_s, w_out, bias=True)
            self.cf_af = Hardswish()
        self.dropout = Dropout(p=dropout_ratio) if dropout_ratio > 0 else None
        self.fc = linear(w_out, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_af(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if self.nl:
            self.cf_af(self.fc1(x))
        x = self.dropout(x) if self.dropout else x
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, num_classes, exp_s, nl):
        exp_s = w_out if nl == 0 else exp_s
        cx = conv2d_cx(cx, w_in, exp_s, 1)
        cx = norm2d_cx(cx, exp_s)
        cx = gap2d_cx(cx, exp_s)
        if nl:
            cx = linear_cx(cx, exp_s, w_out, bias=True)
        cx = linear_cx(cx, w_out, num_classes, bias=True)
        return cx

    def fuse_model(self, include_relu: bool):
        targets = (
            [["conv", "conv_bn", "conv_af"]] if include_relu else [
                ["conv", "conv_bn"]]
        )
        fuse_modules(self, targets, inplace=True)


class MobileNet(Module):
    """"MobileNet model."""

    @staticmethod
    def get_params():
        return {
            "sw": cfg.MN.STEM_W,
            "ds": cfg.MN.DEPTHS,
            "ws": cfg.MN.WIDTHS,
            "exp_sz": cfg.MN.EXP_SIZES,
            "ss": cfg.MN.STRIDES,
            "ks": cfg.MN.KERNELS,
            "nl": cfg.MN.NONLINEARITY,
            "se": cfg.MN.SQUEEZE_AND_EXCITE,
            "hw": cfg.MN.HEAD_W,
            "nc": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self, params=None):
        super(MobileNet, self).__init__()
        p = MobileNet.get_params() if not params else params
        vs = ["sw", "ds", "ws", "exp_sz", "ss", "ks", "nl", "se", "hw", "nc"]
        sw, ds, ws, exp_sz, ss, ks, nl, se, hw, nc = [p[v] for v in vs]
        stage_params = list(zip(ds, ws, exp_sz, ss, ks, nl, se))
        self.stem = StemImageNet(3, sw, nl)
        prev_w = sw
        for i, (d, w, exp_s, stride, ks, nl, se) in enumerate(stage_params):
            stage = MNStage(prev_w, exp_s, stride, w, d, ks, nl, se)
            self.add_module("s{}".format(i + 1), stage)
            prev_w = w
        self.head = MNHead(prev_w, hw, nc, exp_s, nl)
        self.apply(init_weights)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        p = MobileNet.get_params() if not params else params
        vs = ["sw", "ds", "ws", "exp_sz", "ss", "ks", "nl", "se", "hw", "nc"]
        sw, ds, ws, exp_sz, ss, ks, nl, se, hw, nc = [p[v] for v in vs]
        stage_params = list(zip(ds, ws, exp_sz, ss, ks, nl, se))
        cx = StemImageNet.complexity(cx, 3, sw)
        prev_w = sw
        for d, w, exp_s, stride, ks, nl, se in stage_params:
            cx = MNStage.complexity(cx, prev_w, exp_s, stride, w, d, ks, se)
            prev_w = w
        cx = MNHead.complexity(cx, prev_w, hw, nc, exp_s, nl)
        return cx

    def fuse_model(self, include_relu: bool = False):
        self.stem.fuse_model(include_relu)
        for m in self.modules():
            if type(m) == MNStage:
                m.fuse_model(include_relu)
        self.head.fuse_model(include_relu)

    def postprocess_skip(self):
        for mod in self.modules():
            if isinstance(mod, MNStage) and len(mod._modules) > 1:
                fakeq = mod._modules["b1"].lin_proj.activation_post_process
                for _, m in list(mod._modules.items())[1:]:
                    m.lin_proj.activation_post_process = fakeq
