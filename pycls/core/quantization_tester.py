"""Tools for training and testing a model."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pycls.core.benchmark as benchmark
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
from pycls.models.model_quantizer import (
    HistogramShiftObserver,
    MinMaxShiftObserver,
    QuantizedModel,
)
from torch.quantization.observer import HistogramObserver, MinMaxObserver

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.utils.data import DataLoader

logger = logging.get_logger(__name__)


def setup_cpu_env():
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


def model_equivalence(
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


def fuse_network(model: Module):
    fused_model = deepcopy(model)
    fused_model.eval()
    model.eval()
    fused_model.fuse_model(cfg.QUANTIZATION.ACT_FUSION)
    # Model and fused model should be equivalent.
    assert model_equivalence(
        model_1=model,
        model_2=fused_model,
        rtol=1e-02,
        atol=1e-04,
        num_tests=100,
        input_size=(1, 3, 224, 224),
    ), "Fused model is not equivalent to the original model!"
    return fused_model


def setup_cpu_model():
    """Sets up a model and log the results."""
    # Build the model
    model = builders.build_model()
    logger.info("Model:\n{}".format(model)) if cfg.VERBOSE else ()
    # Log model complexity
    logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    return model


@torch.no_grad()
def test_cpu_model_epoch(
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


def calibrate_model(
    model: QuantizedModel, loader: DataLoader, device=torch.device("cpu:0")
):
    model.to(device)
    model.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)


def get_observer(method: str):
    observer = None
    if "min_max" == method:
        observer = MinMaxObserver
    elif "mm_shift" == method:
        observer = MinMaxShiftObserver
    elif "histogram" == method:
        observer = HistogramObserver
    elif "hist_shift" == method:
        observer = HistogramShiftObserver
    else:
        raise AttributeError("Not supported")
    return observer


def quantize_model(model: Module, loader: DataLoader, method: str):
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
    setup_cpu_env()
    # Construct the model
    model = setup_cpu_model()
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
        quantized_model = quantize_model(fused_model, calibration_loader, "min_max")
        logger.info("Min-Max")
        test_cpu_model_epoch(test_loader, quantized_model, test_meter, 0)

    if "mm_shift" in cfg.QUANTIZATION.METHOD:
        quantized_model = quantize_model(fused_model, calibration_loader, "mm_shift")
        logger.info(f"Min-Max Shift Quantization")
        test_cpu_model_epoch(test_loader, quantized_model, test_meter, 0)

    if "histogram" in cfg.QUANTIZATION.METHOD:
        quantized_model = quantize_model(fused_model, calibration_loader, "histogram")
        logger.info(f"Histogram Quantization")
        test_cpu_model_epoch(test_loader, quantized_model, test_meter, 0)

    if "hist_shift" in cfg.QUANTIZATION.METHOD:
        quantized_model = quantize_model(fused_model, calibration_loader, "hist_shift")
        logger.info(f"Histogram Shift Quantization")
        test_cpu_model_epoch(test_loader, quantized_model, test_meter, 0)

    if "float" in cfg.QUANTIZATION.METHOD:
        logger.info("Float32")
        cur_device = torch.cuda.current_device()
        model = model.cuda(device=cur_device)
        trainer.test_epoch(test_loader, model, test_meter, 0)
