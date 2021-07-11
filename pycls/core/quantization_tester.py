"""Tools for training and testing a model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pycls.core.builders as builders
import pycls.core.checkpoint as cp
import pycls.core.config as config
import pycls.core.logging as logging
import pycls.core.meters as meters
import pycls.core.net as net
import pycls.core.trainer as trainer
import pycls.datasets.loader as data_loader
import torch
from pycls.core.config import cfg
from pycls.core.io import pathmgr
from pycls.core.quantization_utils import (
    calibrate_model,
    fuse_network,
    get_observer,
    model2cuda,
    quantize_network_for_qat,
    setup_model,
)
from pycls.quantization.hw_quant_op import QuantOps
from pycls.quantization.quantizer import QuantizedModel
from torch.quantization.observer import HistogramObserver

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.utils.data import DataLoader

logger = logging.get_logger(__name__)


def _setup_cpu_env():
    """Sets up environment for quatized model testing."""
    # Ensure that the output dir exists
    pathmgr.mkdirs(cfg.OUT_DIR)
    # Save the config
    config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log torch versions
    logger.info("PyTorch Version: torch={}".format(torch.__version__))
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg)) if cfg.VERBOSE else ()
    logger.info(logging.dump_log_data(cfg, "cfg", None))


def _setup_cpu_model():
    """Sets up a model and log the results."""
    # Build the model
    model = builders.build_model()
    logger.info("Model:\n{}".format(model)) if cfg.VERBOSE else ()
    # Log model complexity
    logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    return model


@torch.no_grad()
def _test_cpu_model_epoch(
    loader: DataLoader, model: QuantizedModel, meter: meters.TestMeter, cur_epoch: int
):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    meter.reset()
    meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(loader):
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        top1_err, top5_err = top1_err.item(), top5_err.item()
        meter.iter_toc()
        # Update and log stats
        meter.update_stats(top1_err, top5_err, inputs.size(0))
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)


def _quantize_model4ptq(model: Module, loader: DataLoader, method: str):
    quantized_model = QuantizedModel(model_fp32=model)

    observer = get_observer(method)
    quantization_config = torch.quantization.QConfig(
        activation=observer.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_symmetric
        ),
        weight=HistogramObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        ),
    )
    quantized_model.qconfig = quantization_config

    q_model = torch.quantization.quantize(
        model=quantized_model,
        run_fn=calibrate_model,
        run_args=[loader],
        mapping=None,
        inplace=False,
    )

    logger.info(f"Quantized Model:\n {q_model}")
    return q_model


def test_quantized_model():
    """Evaluates a quantized model."""
    # Setup training/testing environment
    _setup_cpu_env()
    # Construct the model
    model = _setup_cpu_model()
    # Load model weights
    cp.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))

    # Create data loaders and meters
    test_loader = data_loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    calibration_loader = data_loader.construct_calibration_loader()

    if ("float") != cfg.QUANTIZATION.METHOD:
        fused_model = fuse_network(model)
    quantized_model = None

    if "min_max" in cfg.QUANTIZATION.METHOD:
        quantized_model = _quantize_model4ptq(
            fused_model, calibration_loader, "min_max"
        )
        logger.info("Min-Max")
        _test_cpu_model_epoch(test_loader, quantized_model, test_meter, 0)

    if "mm_shift" in cfg.QUANTIZATION.METHOD:
        quantized_model = _quantize_model4ptq(
            fused_model, calibration_loader, "mm_shift"
        )
        logger.info(f"Min-Max Shift Quantization")
        _test_cpu_model_epoch(test_loader, quantized_model, test_meter, 0)

    if "avg_mm_shift" in cfg.QUANTIZATION.METHOD:
        quantized_model = _quantize_model4ptq(
            fused_model, calibration_loader, "avg_mm_shift"
        )
        logger.info(f"Moving Average Min-Max Shift Quantization")
        _test_cpu_model_epoch(test_loader, quantized_model, test_meter, 0)

    if "histogram" in cfg.QUANTIZATION.METHOD:
        quantized_model = _quantize_model4ptq(
            fused_model, calibration_loader, "histogram"
        )
        logger.info(f"Histogram Quantization")
        _test_cpu_model_epoch(test_loader, quantized_model, test_meter, 0)

    if "hist_shift" in cfg.QUANTIZATION.METHOD:
        quantized_model = _quantize_model4ptq(
            fused_model, calibration_loader, "hist_shift"
        )
        logger.info(f"Histogram Shift Quantization")
        _test_cpu_model_epoch(test_loader, quantized_model, test_meter, 0)

    if "float" in cfg.QUANTIZATION.METHOD:
        logger.info("Float32")
        cur_device = torch.cuda.current_device()
        model = model.cuda(device=cur_device)
        trainer.test_epoch(test_loader, model, test_meter, 0)


def test_qat_network():
    """Trains the quantized model. Most are copied from 'trainer.py.'"""
    # Setup training/testing environment
    trainer.setup_env()
    # Construct the model, loss_fun, and optimizer
    model = setup_model()
    # Create data loaders and meters
    test_loader = data_loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))

    model, _, _ = quantize_network_for_qat(model)
    if cp.has_checkpoint() or cfg.TEST.WEIGHTS:
        checkpoint_file = (
            cp.get_last_checkpoint() if cp.has_checkpoint() else cfg.TEST.WEIGHTS
        )
        cp.load_checkpoint(checkpoint_file, model)

    logger.info("GPU Results")
    trainer.test_epoch(test_loader, model, test_meter, 0)

    cpu_device = torch.device("cpu")
    model = net.unwrap_model(model).to(cpu_device)
    _convert(model)
    model = model2cuda(model)
    print(model)
    logger.info("MIDAP Results")
    trainer.test_epoch(test_loader, model, test_meter, 0)


def _convert(module):
    swapped_module = {}
    for n, m in module.named_children():
        if type(m) in QuantOps:
            swapped_module[n] = QuantOps[type(m)].from_trained_op(m)
        else:
            _convert(m)

    for n, m in swapped_module.items():
        module._modules[n] = m
