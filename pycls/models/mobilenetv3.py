"""MobileNetV3 models."""

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
    hswish,
)
from torch.nn import Dropout, Module
from torch.nn.quantized import FloatFunctional
from torch.quantization import fuse_modules


class StemImageNet(Module):
    """MobileNetV3 stem for ImageNet: 3x3, BN, AF(Hswish)."""

    def __init__(self, w_in, w_out):
        super(StemImageNet, self).__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(w_out)
        self.af = hswish(w_in)

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
        targets = [["conv", "bn", "af"]] if include_relu else [["conv", "bn"]]
        fuse_modules(self, targets, inplace=True)


class MBConv(Module):
    """MobileNetV3 inverted bottleneck block"""

    def __init__(self, w_in, exp_r, stride, w_out):
        # Expansion, 3x3 depthwise, BN, AF, 1x1 pointwise, BN, skip_connection
        super(MBConv, self).__init__()
        self.stride = stride
        assert stride in [1, 2]  # stride must be 1 or 2
        self.exp = None
        w_exp = int(w_in * exp_r)  # expand channel using expansion factor(exp_r)
        if w_exp != w_in:  # skip if exp_r is 1
            self.exp = conv2d(w_in, w_exp, 1)
            self.exp_bn = norm2d(w_exp)
            self.exp_af = activation()
        # depthwise
        self.dwise = conv2d(w_exp, w_exp, 3, stride=stride, groups=w_exp)
        self.dwise_bn = norm2d(w_exp)
        self.dwise_af = activation()
        # pointwise
        self.lin_proj = conv2d(w_exp, w_out, 1)
        self.lin_proj_bn = norm2d(w_out)
        self.use_res_connect = (
            self.stride == 1 and w_in == w_out
        )  # check skip connection
        if self.use_res_connect:
            self.skip_add = FloatFunctional()

    def forward(self, x):
        f_x = self.exp_af(self.exp_bn(self.exp(x))) if self.exp else x
        f_x = self.dwise_af(self.dwise_bn(self.dwise(f_x)))
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.use_res_connect:
            f_x = self.skip_add.add(x, f_x)
        return f_x

    @staticmethod
    def complexity(cx, w_in, exp_r, stride, w_out):
        w_exp = int(w_in * exp_r)  # expand channel using expansion factor
        if w_exp != w_in:
            cx = conv2d_cx(cx, w_in, w_exp, 1)
            cx = norm2d_cx(cx, w_exp)
        # depthwise
        cx = conv2d_cx(cx, w_exp, w_exp, 3, stride=stride, groups=w_exp)
        cx = norm2d_cx(cx, w_exp)
        # pointwise
        cx = conv2d_cx(cx, w_exp, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx

    def fuse_model(self, include_relu: bool):
        targets = (
            [["dwise", "dwise_bn", "dwise_af"], ["lin_proj", "lin_proj_bn"]]
            if include_relu
            else [["dwise", "dwise_bn"], ["lin_proj", "lin_proj_bn"]]
        )
        if self.exp:
            targets.append(
                ["exp", "exp_bn", "exp_af"] if include_relu else ["exp", "exp_bn"]
            )
        fuse_modules(self, targets, inplace=True)


class MNV3Stage(Module):
    """MobileNetV3 stage."""

    def __init__(self, w_in, exp_r, stride, w_out, d):
        super(MNV3Stage, self).__init__()

        for i in range(d):  # d는 layer의 반복 횟수
            stride = stride if i == 0 else 1
            block = MBConv(w_in, exp_r, stride, w_out)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, exp_r, stride, w_out, d):
        for i in range(d):
            stride = stride if i == 0 else 1
            cx = MBConv.complexity(cx, w_in, exp_r, stride, w_out)
            stride, w_in = 1, w_out
            return cx

    def fuse_model(self, include_relu: bool):
        for m in self.modules():
            if type(m) == MBConv:
                m.fuse_model(include_relu)


class MNV3Head(Module):
    """MobileNetV3 head: 1x1, BN, AF(ReLU6), AvgPool, Dropout, FC."""

    def __init__(self, w_in, w_out, num_classes):
        super(MNV3Head, self).__init__()
        dropout_ratio = cfg.MNV2.DROPOUT_RATIO
        self.conv = conv2d(w_in, w_out, 1)
        self.conv_bn = norm2d(w_out)
        self.conv_af = activation()
        self.avg_pool = gap2d(w_out)
        # classifier
        self.dropout = Dropout(p=dropout_ratio) if dropout_ratio > 0 else None
        self.fc = linear(w_out, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_af(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = self.conv_af(self.conv_bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x) if self.dropout else x
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, num_classes):
        cx = conv2d_cx(cx, w_in, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        cx = gap2d_cx(cx, w_out)
        cx = linear_cx(cx, w_out, num_classes, bias=True)
        return cx

    def fuse_model(self, include_relu: bool):
        targets = (
            [["conv", "conv_bn", "conv_af"]] if include_relu else [["conv", "conv_bn"]]
        )
        fuse_modules(self, targets, inplace=True)


class MobileNetV3(Module):
    """"MobileNetV3 model."""

    @staticmethod
    def get_params():
        return {
            "sw": cfg.MNV2.STEM_W,
            "ds": cfg.MNV2.DEPTHS,
            "ws": cfg.MNV2.WIDTHS,
            "exp_rs": cfg.MNV2.EXP_RATIOS,
            "ss": cfg.MNV2.STRIDES,
            "hw": cfg.MNV2.HEAD_W,
            "nc": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self, params=None):
        super(MobileNetV3, self).__init__()
        p = MobileNetV3.get_params() if not params else params
        p["nl"] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        vs = ["sw", "ds", "ws", "exp_rs", "ss", "hw", "nc", "nl"]

        sw, ds, ws, exp_rs, ss, hw, nc, nl = [p[v] for v in vs]
        stage_params = list(zip(ds, ws, exp_rs, ss))
        # sw = 16
        self.stem = StemImageNet(3, sw)
        prev_w = sw
        for i, (d, w, exp_r, stride) in enumerate(stage_params):
            stage = MNV3Stage(prev_w, exp_r, stride, w, d)
            self.add_module("s{}".format(i + 1), stage)
            prev_w = w
        self.head = MNV3Head(prev_w, hw, nc)
        self.apply(init_weights)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        p = MobileNetV3.get_params() if not params else params
        vs = ["sw", "ds", "ws", "exp_rs", "ss", "hw", "nc"]
        sw, ds, ws, exp_rs, ss, hw, nc = [p[v] for v in vs]
        stage_params = list(zip(ds, ws, exp_rs, ss))
        cx = StemImageNet.complexity(cx, 3, sw)
        prev_w = sw
        for d, w, exp_r, stride in stage_params:
            cx = MNV3Stage.complexity(cx, prev_w, exp_r, stride, w, d)
            prev_w = w
        cx = MNV3Head.complexity(cx, prev_w, hw, nc)
        return cx

    def fuse_model(self, include_relu: bool = False):
        self.stem.fuse_model(include_relu)
        for m in self.modules():
            if type(m) == MNV3Stage:
                m.fuse_model(include_relu)
        self.head.fuse_model(include_relu)

    def postprocess_skip(self):
        for mod in self.modules():
            if isinstance(mod, MNV3Stage) and len(mod._modules) > 1:
                fakeq = mod._modules["b1"].lin_proj.activation_post_process
                for _, m in list(mod._modules.items())[1:]:
                    m.lin_proj.activation_post_process = fakeq
