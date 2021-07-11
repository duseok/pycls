from __future__ import annotations

from typing import TYPE_CHECKING

import pycls.core.checkpoint as cp
import pycls.core.config as config
import pycls.core.logging as logging
import pycls.core.meters as meters
import pycls.core.net as net
import pycls.core.optimizer as optim
import pycls.core.trainer as trainer
import pycls.datasets.loader as data_loader
import torch
import torch.cuda.amp as amp
from pycls.core.config import cfg
from pycls.core.io import pathmgr
from pycls.core.quantization_utils import (
    calibrate_model,
    model2cuda,
    quantize_network_for_qat,
    setup_model,
)
from torch.optim.optimizer import Optimizer

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.utils.data import DataLoader

logger = logging.get_logger(__name__)


def _get_info_from_checkpoint4qat(file):
    err_str = "Checkpoint '{}' not found"
    assert pathmgr.exists(file), err_str.format(file)
    with pathmgr.open(file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    bn_start_epoch, ft_start_epoch = 0, 0
    if "step" in checkpoint:
        step = checkpoint["step"]
        start_epoch = checkpoint["epoch"] + 1
        if step == "bn_stabilization":
            bn_start_epoch = start_epoch
        else:
            ft_start_epoch = start_epoch
    else:
        logger.warning(
            f"This checkpoint does not contain 'step.' Load weights and start from stabilizing BN."
        )
        step = "bn_stabilization"

    return step, bn_start_epoch, ft_start_epoch


def _setup_teacher():
    cfg.defrost()
    config.load_cfg_fom_args(description="Teacher model", cfg_file=cfg.TRAIN.TEACHER)
    config.assert_and_infer_cfg()
    cfg.freeze()
    teacher = setup_model()
    assert cfg.TRAIN.TEACHER_WEIGHTS != ""
    cp.load_checkpoint(cfg.TRAIN.TEACHER_WEIGHTS, teacher, None)
    logger.info(
        "Loaded initial teacher weights from: {}".format(cfg.TRAIN.TEACHER_WEIGHTS)
    )
    return model2cuda(teacher)


def _restore_cfg():
    cfg.defrost()
    config.reset_cfg()
    config.load_cfg_fom_args("Restore a network configuration.")
    config.assert_and_infer_cfg()
    cfg.freeze()


def _categorize_params(model):
    scale = []
    bnbias = []
    weights = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif name.endswith(".scale"):
            scale.append(param)
        elif len(param.shape) == 1 or name.endswith(".bias"):
            bnbias.append(param)
        else:
            weights.append(param)
    return weights, bnbias, scale


def _stabilize_bn(
    model: Module,
    ema: Module,
    teacher: Module,
    start_epoch: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fun: Module,
    optimizer: Optimizer,
    scaler: amp.GradScaler,
    train_meter: meters.TrainMeter,
    test_meter: meters.TrainMeter,
    ema_meter: meters.TrainMeter,
):
    max_finetune_epoch = cfg.OPTIM.MAX_EPOCH
    cfg.defrost()
    cfg.OPTIM.MAX_EPOCH = cfg.QUANTIZATION.QAT.BN_STABILIZATION_EPOCH
    cfg.freeze()
    logger.info("Start BN stabilization (Epoch: {})".format(start_epoch + 1))
    _run_qat_newtork(
        model,
        ema,
        teacher,
        start_epoch,
        train_loader,
        test_loader,
        loss_fun,
        optimizer,
        scaler,
        train_meter,
        test_meter,
        ema_meter,
        prefix="bn_stabilization_",
    )
    cfg.defrost()
    cfg.OPTIM.MAX_EPOCH = max_finetune_epoch
    cfg.freeze()


def _enable_bias_quant(module):
    from pycls.quantization.quant_op import QConv2d, QLinear

    for _, m in module.named_children():
        if isinstance(m, QConv2d) or isinstance(m, QLinear):
            m.set_quant_bias(True)
        else:
            _enable_bias_quant(m)


def train_qat_network():
    """Trains the quantized model. Most are copied from 'trainer.py.'"""
    # Setup training/testing environment
    trainer.setup_env()
    # Construct the model, loss_fun, and optimizer
    model = setup_model()
    # Load checkpoint or initial weights
    if cfg.QUANTIZATION.QAT.FP32_WEIGHTS:
        cp.load_checkpoint(cfg.QUANTIZATION.QAT.FP32_WEIGHTS, model, None)
        logger.info(
            "Loaded initial fp32 weights from: {}".format(
                cfg.QUANTIZATION.QAT.FP32_WEIGHTS
            )
        )
    # Create data loaders and meters
    train_loader = data_loader.construct_train_loader()
    test_loader = data_loader.construct_test_loader()
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    ema_meter = meters.TestMeter(len(test_loader), "test_ema")

    model, ema, loss_fun = quantize_network_for_qat(model)
    bn_start_epoch, ft_start_epoch = 0, 0
    start_step = "bn_stabilization"
    checkpoint_file = None
    if cp.has_checkpoint() or cfg.TRAIN.WEIGHTS:
        checkpoint_file = (
            cp.get_last_checkpoint() if cp.has_checkpoint() else cfg.TRAIN.WEIGHTS
        )
        start_step, bn_start_epoch, ft_start_epoch = _get_info_from_checkpoint4qat(
            checkpoint_file
        )

    teacher = None
    if str.lower(cfg.TRAIN.TEACHER) != "":
        teacher = _setup_teacher()
        _restore_cfg()

    # Create a GradScaler for mixed precision training
    scaler = amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    if not checkpoint_file:
        logger.info("Start Calibration")
        calibrate_model(model, train_loader, model.device, 1000 / cfg.TRAIN.BATCH_SIZE)
    model.apply(torch.quantization.disable_observer)
    ema.apply(torch.quantization.disable_observer)

    if cfg.QUANTIZATION.QAT.TRAIN_SHIFT_BIAS_QUANTIZATION:
        _enable_bias_quant(model)

    train_params = _categorize_params(model)
    if start_step == "bn_stabilization":
        stabilize_opt = optim.construct_optimizer(model, train_params, False)
        if checkpoint_file:
            cp.load_checkpoint(checkpoint_file, model, ema, stabilize_opt)
        _stabilize_bn(
            model,
            ema,
            teacher,
            bn_start_epoch,
            train_loader,
            test_loader,
            loss_fun,
            stabilize_opt,
            scaler,
            train_meter,
            test_meter,
            ema_meter,
        )

    logger.info("Start finetuing (Epoch: {})".format(ft_start_epoch + 1))
    optimizer = optim.construct_optimizer(model, train_params)
    if start_step == "default":
        cp.load_checkpoint(checkpoint_file, model, ema, optimizer)
    _run_qat_newtork(
        model,
        ema,
        teacher,
        ft_start_epoch,
        train_loader,
        test_loader,
        loss_fun,
        optimizer,
        scaler,
        train_meter,
        test_meter,
        ema_meter,
        bn_freeze=(cfg.QUANTIZATION.QAT.BN_TRAIN_EPOCH != -1),
    )


def _run_qat_newtork(
    model: Module,
    ema: Module,
    teacher: Module,
    start_epoch: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fun: Module,
    optimizer: Optimizer,
    scaler: amp.GradScaler,
    train_meter: meters.TrainMeter,
    test_meter: meters.TrainMeter,
    ema_meter: meters.TrainMeter,
    bn_freeze=False,
    prefix="",
):
    # Perform the training loop
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        params = (train_loader, model, ema, loss_fun, optimizer, scaler, train_meter)
        trainer.train_epoch(*params, cur_epoch, teacher)

        if bn_freeze and cur_epoch <= cfg.QUANTIZATION.QAT.BN_TRAIN_EPOCH - 1:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            ema.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
            net.compute_precise_bn_stats(ema, train_loader)

        # Evaluate the model
        trainer.test_epoch(test_loader, model, test_meter, cur_epoch)
        trainer.test_epoch(test_loader, ema, ema_meter, cur_epoch)

        test_err = test_meter.get_epoch_stats(cur_epoch)["top1_err"]
        ema_err = ema_meter.get_epoch_stats(cur_epoch)["top1_err"]
        # Save a checkpoint
        file = cp.save_checkpoint(
            model, ema, optimizer, cur_epoch, test_err, ema_err, prefix
        )
        logger.info("Wrote checkpoint to: {}".format(file))
