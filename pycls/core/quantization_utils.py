from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pycls.core.builders as builders
import pycls.core.logging as logging
import torch
import torch.nn as nn
from pycls.core.config import cfg
from pycls.core.setup import model2cuda
from pycls.quantization.hw_quant_op import QuantOps
from pycls.quantization.quant_op import QConv2d, QConvBn2d, QLinear
from pycls.quantization.quantizer import QuantizedModel
from pycls.quantization.shift_fake_quantizer import ShiftFakeQuantize
from pycls.quantization.shift_observer import (
    HistogramShiftObserver,
    MinMaxShiftObserver,
    MovingAvgMinMaxShiftObserver,
)
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch.nn.quantized.modules.functional_modules import FloatFunctional
from torch.quantization.observer import HistogramObserver, MinMaxObserver

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.utils.data import DataLoader

logger = logging.get_logger(__name__)


def get_observer(method: str):
    observer = None
    if "min_max" == method:
        observer = MinMaxObserver
    elif "mm_shift" == method:
        observer = MinMaxShiftObserver
    elif "avg_mm_shift" == method:
        observer = MovingAvgMinMaxShiftObserver
    elif "histogram" == method:
        observer = HistogramObserver
    elif "hist_shift" == method:
        observer = HistogramShiftObserver
    else:
        raise AttributeError("Not supported")
    return observer


def _model_equivalence(
    model_1: Module,
    model_2: Module,
    rtol=1e-05,
    atol=1e-08,
    num_tests=100,
    input_size=(1, 3, 32, 32),
):
    for _ in range(num_tests):
        x = torch.rand(size=input_size)
        y1 = model_1(x)
        y2 = model_2(x)
        if not torch.allclose(y1, y2, rtol=rtol, atol=atol, equal_nan=False):
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True


def fuse_network(model: Module, with_bn=False, debug=False):
    fused_model = deepcopy(model)
    if with_bn:
        fused_model.train()
    else:
        fused_model.eval()
    fused_model.fuse_model(cfg.QUANTIZATION.ACT_FUSION)
    # Model and fused model should be equivalent.
    if debug:
        model.eval()
        fused_model.eval()
        assert _model_equivalence(
            model_1=model,
            model_2=fused_model,
            rtol=1e-02,
            atol=1e-04,
            num_tests=100,
            input_size=(1, 3, 224, 224),
        ), "Fused model is not equivalent to the original model!"
    return fused_model


@torch.no_grad()
def calibrate_model(model: QuantizedModel, loader: DataLoader, use_cuda=True):
    model.eval()
    for inputs, _ in loader:
        if use_cuda:
            inputs = inputs.cuda()
        _ = model(inputs)


def _quantize_model4qat(model: Module, method: str):
    import numpy as np
    from torch.nn.intrinsic.modules.fused import ConvBn2d
    from torch.quantization.quantization_mappings import get_default_qat_module_mappings

    quantized_model = QuantizedModel(model_fp32=model)

    observer = get_observer(method)
    quantization_config = torch.quantization.QConfig(
        activation=ShiftFakeQuantize.with_args(
            observer=observer,
            quant_min=0,
            quant_max=int(np.exp2(cfg.QUANTIZATION.QAT.ACT_BITWIDTH) - 1),
            dtype=torch.quint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,
        ),
        weight=ShiftFakeQuantize.with_args(
            observer=HistogramShiftObserver,
            quant_min=-int(np.exp2(cfg.QUANTIZATION.QAT.WEIGHT_BITWIDTH - 1)),
            quant_max=int(np.exp2(cfg.QUANTIZATION.QAT.WEIGHT_BITWIDTH - 1) - 1),
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,
        ),
    )
    quantized_model.qconfig = quantization_config

    mapping = get_default_qat_module_mappings()
    if cfg.QUANTIZATION.QAT.TRAIN_SHIFT_BIAS_QUANTIZATION:
        mapping[nn.Conv2d] = QConv2d
        mapping[ConvBn2d] = QConvBn2d
        mapping[nn.Linear] = QLinear

    quantized_model = torch.quantization.prepare_qat(
        quantized_model, mapping, inplace=True
    )

    if cfg.QUANTIZATION.QAT.TRAIN_SAME_SCALE4SKIP:
        quantized_model.postprocess_skip()
    return quantized_model


def _copy_fake_quant(src: QuantizedModel, dest: QuantizedModel):
    for s, d in zip(src.modules(), dest.modules()):
        if isinstance(s, ShiftFakeQuantize) and isinstance(d, ShiftFakeQuantize):
            d.activation_post_process = s.activation_post_process
            d.zero_point = s.zero_point


def get_last_postprocess_before(model: Module, module: Module):
    rev_modules = list(model.modules())
    for mod in reversed(rev_modules[: rev_modules.index(module)]):
        if isinstance(mod, (QConv2d, QConvBn2d, QLinear, FloatFunctional)):
            return mod.activation_post_process


def quantize_network_for_qat(model: QuantizedModel):
    model.train()
    model = fuse_network(model, cfg.QUANTIZATION.QAT.WITH_BN)

    assert (
        len(cfg.QUANTIZATION.METHOD) == 1
    ), "When testing QAT, only one quantization method is supported."
    model = _quantize_model4qat(model, cfg.QUANTIZATION.METHOD[0])
    if cfg.QUANTIZATION.QAT.TRAIN_SHIFT_AVG_POOL:
        head = model.model_fp32.head
        prev_post_process = get_last_postprocess_before(model.model_fp32, head.avg_pool)
        head.avg_pool = QuantOps[AdaptiveAvgPool2d].from_trained_op(prev_post_process)

    model = model2cuda(model)
    ema = deepcopy(model)
    _copy_fake_quant(model, ema)
    loss_fun = builders.build_loss_fun().cuda()

    logger.info(f"QAT Model:\n {model}")
    logger.info(f"QAT EMA Model:\n {ema}")
    return model, ema, loss_fun
