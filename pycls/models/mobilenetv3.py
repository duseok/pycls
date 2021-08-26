# """MobileNetV3 models."""

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


"""MobileNetV3 models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import numpy as np

import tensorflow.compat.v1 as tf
import tf_slim as slim

# from nets.mobilenet import conv_blocks as ops
# from nets.mobilenet import mobilenet as lib

_Op = collections.namedtuple('Op', ['op', 'params', 'multiplier_func'])


def op(opfunc, multiplier_func=depth_multiplier, **params):
    multiplier = params.pop('multiplier_transform', multiplier_func)
    return _Op(opfunc, params=params, multiplier_func=multiplier)


expand_input = ops.expand_input_by_factor


def expanded_conv(input_tensor,
                  num_outputs,
                  expansion_size=expand_input_by_factor(6),
                  stride=1,
                  rate=1,
                  kernel_size=(3, 3),
                  residual=True,
                  normalizer_fn=None,
                  split_projection=1,
                  split_expansion=1,
                  split_divisible_by=8,
                  expansion_transform=None,
                  depthwise_location='expansion',
                  depthwise_channel_multiplier=1,
                  endpoints=None,
                  use_explicit_padding=False,
                  padding='SAME',
                  inner_activation_fn=None,
                  depthwise_activation_fn=None,
                  project_activation_fn=tf.identity,
                  depthwise_fn=slim.separable_conv2d,
                  expansion_fn=split_conv,
                  projection_fn=split_conv,
                  scope=None):
    """Depthwise Convolution Block with expansion.
    Builds a composite convolution that has the following structure
    expansion (1x1) -> depthwise (kernel_size) -> projection (1x1)
    Args:
      input_tensor: input
      num_outputs: number of outputs in the final layer.
      expansion_size: the size of expansion, could be a constant or a callable.
        If latter it will be provided 'num_inputs' as an input. For forward
        compatibility it should accept arbitrary keyword arguments.
        Default will expand the input by factor of 6.
      stride: depthwise stride
      rate: depthwise rate
      kernel_size: depthwise kernel
      residual: whether to include residual connection between input
        and output.
      normalizer_fn: batchnorm or otherwise
      split_projection: how many ways to split projection operator
        (that is conv expansion->bottleneck)
      split_expansion: how many ways to split expansion op
        (that is conv bottleneck->expansion) ops will keep depth divisible
        by this value.
      split_divisible_by: make sure every split group is divisible by this number.
      expansion_transform: Optional function that takes expansion
        as a single input and returns output.
      depthwise_location: where to put depthwise covnvolutions supported
        values None, 'input', 'output', 'expansion'
      depthwise_channel_multiplier: depthwise channel multiplier:
      each input will replicated (with different filters)
      that many times. So if input had c channels,
      output will have c x depthwise_channel_multpilier.
      endpoints: An optional dictionary into which intermediate endpoints are
        placed. The keys "expansion_output", "depthwise_output",
        "projection_output" and "expansion_transform" are always populated, even
        if the corresponding functions are not invoked.
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      padding: Padding type to use if `use_explicit_padding` is not set.
      inner_activation_fn: activation function to use in all inner convolutions.
      If none, will rely on slim default scopes.
      depthwise_activation_fn: activation function to use for deptwhise only.
        If not provided will rely on slim default scopes. If both
        inner_activation_fn and depthwise_activation_fn are provided,
        depthwise_activation_fn takes precedence over inner_activation_fn.
      project_activation_fn: activation function for the project layer.
      (note this layer is not affected by inner_activation_fn)
      depthwise_fn: Depthwise convolution function.
      expansion_fn: Expansion convolution function. If use custom function then
        "split_expansion" and "split_divisible_by" will be ignored.
      projection_fn: Projection convolution function. If use custom function then
        "split_projection" and "split_divisible_by" will be ignored.
      scope: optional scope.
    Returns:
      Tensor of depth num_outputs
    Raises:
      TypeError: on inval
    """
    conv_defaults = {}
    dw_defaults = {}
    if inner_activation_fn is not None:
        conv_defaults['activation_fn'] = inner_activation_fn
        dw_defaults['activation_fn'] = inner_activation_fn
    if depthwise_activation_fn is not None:
        dw_defaults['activation_fn'] = depthwise_activation_fn
    # pylint: disable=g-backslash-continuation
    with tf.variable_scope(scope, default_name='expanded_conv') as s, \
            tf.name_scope(s.original_name_scope), \
            slim.arg_scope((slim.conv2d,), **conv_defaults), \
            slim.arg_scope((slim.separable_conv2d,), **dw_defaults):
        prev_depth = input_tensor.get_shape().as_list()[3]
        if depthwise_location not in [None, 'input', 'output', 'expansion']:
            raise TypeError('%r is unknown value for depthwise_location' %
                            depthwise_location)
        if use_explicit_padding:
            if padding != 'SAME':
                raise TypeError('`use_explicit_padding` should only be used with '
                                '"SAME" padding.')
            padding = 'VALID'
        depthwise_func = functools.partial(
            depthwise_fn,
            num_outputs=None,
            kernel_size=kernel_size,
            depth_multiplier=depthwise_channel_multiplier,
            stride=stride,
            rate=rate,
            normalizer_fn=normalizer_fn,
            padding=padding,
            scope='depthwise')
        # b1 -> b2 * r -> b2
        #   i -> (o * r) (bottleneck) -> o
        input_tensor = tf.identity(input_tensor, 'input')
        net = input_tensor

        if depthwise_location == 'input':
            if use_explicit_padding:
                net = _fixed_padding(net, kernel_size, rate)
            net = depthwise_func(net, activation_fn=None)
            net = tf.identity(net, name='depthwise_output')
            if endpoints is not None:
                endpoints['depthwise_output'] = net

        if callable(expansion_size):
            inner_size = expansion_size(num_inputs=prev_depth)
        else:
            inner_size = expansion_size

        if inner_size > net.shape[3]:
            if expansion_fn == split_conv:
                expansion_fn = functools.partial(
                    expansion_fn,
                    num_ways=split_expansion,
                    divisible_by=split_divisible_by,
                    stride=1)
            net = expansion_fn(
                net,
                inner_size,
                scope='expand',
                normalizer_fn=normalizer_fn)
            net = tf.identity(net, 'expansion_output')
            if endpoints is not None:
                endpoints['expansion_output'] = net

        if depthwise_location == 'expansion':
            if use_explicit_padding:
                net = _fixed_padding(net, kernel_size, rate)
            net = depthwise_func(net)
            net = tf.identity(net, name='depthwise_output')
            if endpoints is not None:
                endpoints['depthwise_output'] = net

        if expansion_transform:
            net = expansion_transform(
                expansion_tensor=net, input_tensor=input_tensor)
        # Note in contrast with expansion, we always have
        # projection to produce the desired output size.
        if projection_fn == split_conv:
            projection_fn = functools.partial(
                projection_fn,
                num_ways=split_projection,
                divisible_by=split_divisible_by,
                stride=1)
        net = projection_fn(
            net,
            num_outputs,
            scope='project',
            normalizer_fn=normalizer_fn,
            activation_fn=project_activation_fn)
        if endpoints is not None:
            endpoints['projection_output'] = net
        if depthwise_location == 'output':
            if use_explicit_padding:
                net = _fixed_padding(net, kernel_size, rate)
            net = depthwise_func(net, activation_fn=None)
            net = tf.identity(net, name='depthwise_output')
            if endpoints is not None:
                endpoints['depthwise_output'] = net

        if callable(residual):  # custom residual
            net = residual(input_tensor=input_tensor, output_tensor=net)
        elif (residual and
              # stride check enforces that we don't add residuals when spatial
              # dimensions are None
              stride == 1 and
              # Depth matches
              net.get_shape().as_list()[3] ==
              input_tensor.get_shape().as_list()[3]):
            net += input_tensor
        return tf.identity(net, name='output')


# Squeeze Excite with all parameters filled-in, we use hard-sigmoid
# for gating function and relu for inner activation function.
squeeze_excite = functools.partial(
    ops.squeeze_excite, squeeze_factor=4,
    inner_activation_fn=tf.nn.relu,
    gating_fn=lambda x: tf.nn.relu6(x+3)*0.16667)

# Wrap squeeze excite op as expansion_transform that takes
# both expansion and input tensor.


def _se4(expansion_tensor, input_tensor): return squeeze_excite(
    expansion_tensor)


def hard_swish(x):
    with tf.name_scope('hard_swish'):
        return x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)


def reduce_to_1x1(input_tensor, default_size=7, **kwargs):
    h, w = input_tensor.shape.as_list()[1:3]
    if h is not None and w == h:
        k = [h, h]
    else:
        k = [default_size, default_size]
    return slim.avg_pool2d(input_tensor, kernel_size=k, **kwargs)


def mbv3_op(ef, n, k, s=1, act=tf.nn.relu, se=None, **kwargs):
    """Defines a single Mobilenet V3 convolution block.
    Args:
      ef: expansion factor
      n: number of output channels
      k: stride of depthwise
      s: stride
      act: activation function in inner layers
      se: squeeze excite function.
      **kwargs: passed to expanded_conv
    Returns:
      An object (lib._Op) for inserting in conv_def, representing this operation.
    """
    return op(
        ops.expanded_conv,
        expansion_size=expand_input(ef),
        kernel_size=(k, k),
        stride=s,
        num_outputs=n,
        inner_activation_fn=act,
        expansion_transform=se,
        **kwargs)


def mbv3_fused(ef, n, k, s=1, **kwargs):
    """Defines a single Mobilenet V3 convolution block.
    Args:
      ef: expansion factor
      n: number of output channels
      k: stride of depthwise
      s: stride
      **kwargs: will be passed to mbv3_op
    Returns:
      An object (lib._Op) for inserting in conv_def, representing this operation.
    """
    expansion_fn = functools.partial(slim.conv2d, kernel_size=k, stride=s)
    return mbv3_op(
        ef,
        n,
        k=1,
        s=s,
        depthwise_location=None,
        expansion_fn=expansion_fn,
        **kwargs)


mbv3_op_se = functools.partial(mbv3_op, se=_se4)


DEFAULTS = {
    (ops.expanded_conv,):
        dict(
            normalizer_fn=slim.batch_norm,
            residual=True),
    (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
        'normalizer_fn': slim.batch_norm,
        'activation_fn': tf.nn.relu,
    },
    (slim.batch_norm,): {
        'center': True,
        'scale': True
    },
}

DEFAULTS_GROUP_NORM = {
    (ops.expanded_conv,): dict(normalizer_fn=slim.group_norm, residual=True),
    (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
        'normalizer_fn': slim.group_norm,
        'activation_fn': tf.nn.relu,
    },
    (slim.group_norm,): {
        'groups': 8
    },
}


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3, self).__init__()

        self.bneck = nn.Sequential(
            op(slim.conv2d, stride=2, num_outputs=16, kernel_size=(3, 3),
               activation_fn=hard_swish),
            mbv3_op(ef=1, n=16, k=3),
            mbv3_op(ef=4, n=24, k=3, s=2),
            mbv3_op(ef=3, n=24, k=3, s=1),
            mbv3_op_se(ef=3, n=40, k=5, s=2),
            mbv3_op_se(ef=3, n=40, k=5, s=1),
            mbv3_op_se(ef=3, n=40, k=5, s=1),
            mbv3_op(ef=6, n=80, k=3, s=2, act=hard_swish),
            mbv3_op(ef=2.5, n=80, k=3, s=1, act=hard_swish),
            mbv3_op(ef=184/80., n=80, k=3, s=1, act=hard_swish),
            mbv3_op(ef=184/80., n=80, k=3, s=1, act=hard_swish),
            mbv3_op_se(ef=6, n=112, k=3, s=1, act=hard_swish),
            mbv3_op_se(ef=6, n=112, k=3, s=1, act=hard_swish),
            mbv3_op_se(ef=6, n=160, k=5, s=2, act=hard_swish),
            mbv3_op_se(ef=6, n=160, k=5, s=1, act=hard_swish),
            mbv3_op_se(ef=6, n=160, k=5, s=1, act=hard_swish),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=960,
               activation_fn=hard_swish),
            op(reduce_to_1x1, default_size=7, stride=1, padding='VALID'),
            op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280,
               normalizer_fn=None, activation_fn=hard_swish)
        )

    def forward(self, x):
        out = self.bneck(x)
        return out
