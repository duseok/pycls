from __future__ import annotations

from typing import TYPE_CHECKING

import pycls.core.checkpoint as cp
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
from pycls.core.quantization_utils import calibrate_model, quantize_network_for_qat
from pycls.core.setup import (
    model2cuda,
    restore_cfg,
    setup_env,
    setup_model,
    setup_teacher,
)
from pycls.quantization.quant_op import QConvBn2d
from pycls.quantization.scale_activation import get_scale_act
from pycls.utils import static_vars
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

    if "step" in checkpoint:
        step = checkpoint["step"]
        start_epoch = checkpoint["epoch"] + 1
    else:
        logger.warning(
            f"This checkpoint does not contain 'step.' Load weights and start from stabilizing BN."
        )
        step = "1_bn_stabilization"
        start_epoch = 0

    return step, start_epoch


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
    prefix: str,
):
    max_finetune_epoch = cfg.OPTIM.MAX_EPOCH
    warmup_epochs = cfg.OPTIM.WARMUP_EPOCHS
    cfg.defrost()
    cfg.OPTIM.MAX_EPOCH = cfg.QUANTIZATION.QAT.BN_STABILIZATION_EPOCH
    cfg.OPTIM.WARMUP_EPOCHS = cfg.QUANTIZATION.QAT.BN_STAB_WARMUP_EPOCH
    train_meter.max_iter = cfg.OPTIM.MAX_EPOCH * train_meter.epoch_iters
    cfg.freeze()
    _train_qat_newtork(
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
        prefix=prefix,
    )
    cfg.defrost()
    cfg.OPTIM.MAX_EPOCH = max_finetune_epoch
    cfg.OPTIM.WARMUP_EPOCHS = warmup_epochs
    train_meter.max_iter = cfg.OPTIM.MAX_EPOCH * train_meter.epoch_iters
    cfg.freeze()


def _enable_bias_quant(module):
    from pycls.quantization.quant_op import QConv2d, QLinear

    for _, m in module.named_children():
        if isinstance(m, QConv2d) or isinstance(m, QLinear):
            m.set_quant_bias(True)
        else:
            _enable_bias_quant(m)


def _correct_quant_param(module):
    from pycls.quantization.shift_fake_quantizer import ShiftFakeQuantize

    act_bitwidth = cfg.QUANTIZATION.QAT.ACT_BITWIDTH
    weight_bitwidth = cfg.QUANTIZATION.QAT.WEIGHT_BITWIDTH

    def _correct(s, orig_w, new_w):
        scale_act = get_scale_act()
        _s = scale_act.apply(s) * torch.exp2(orig_w - new_w)
        return scale_act.inverse(_s)

    for _, m in module.named_children():
        if (
            isinstance(m, ShiftFakeQuantize)
            and m.zero_point == 0
            and m.bitwidth != weight_bitwidth
        ):
            m.scale.data = _correct(m.scale.data, m.bitwidth, weight_bitwidth)
            m.bitwidth.copy_(torch.tensor([weight_bitwidth]))
        elif (
            isinstance(m, ShiftFakeQuantize)
            and m.zero_point != 0
            and m.bitwidth != act_bitwidth
        ):
            m.scale.data = _correct(m.scale.data, m.bitwidth, act_bitwidth)
            m.zero_point.copy_(torch.tensor([m.quant_max // 2]))
            m.bitwidth.copy_(torch.tensor([act_bitwidth]))

        if not isinstance(m, ShiftFakeQuantize):
            _correct_quant_param(m)


def _fuse_qat_model(module: Module):
    swapped_module = {}
    for n, m in module.named_children():
        if isinstance(m, QConvBn2d):
            swapped_module[n] = m.fuse_module()
        else:
            _fuse_qat_model(m)

    for n, m in swapped_module.items():
        module._modules[n] = m


def _load_checkpoint(checkpoint_file, model, ema, opt):
    cp.load_checkpoint(checkpoint_file, model, ema, opt)
    model = net.unwrap_model(model)
    ema = net.unwrap_model(ema)
    _correct_quant_param(model)
    _correct_quant_param(ema)
    if cfg.QUANTIZATION.QAT.WITH_BN and cfg.QUANTIZATION.QAT.FOLDING_BN:
        _fuse_qat_model(model)
        _fuse_qat_model(ema)
    model = model2cuda(model)
    ema = model2cuda(ema)
    return model, ema


def train_qat_network():
    """Trains the quantized model. Most are copied from 'trainer.py.'"""
    # Setup training/testing environment
    setup_env()
    # Construct the model, loss_fun, and optimizer
    model = setup_model(False)
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
    start_epoch = 0
    start_step = "1_bn_stabilization"
    checkpoint_file = None
    if cp.has_checkpoint():
        checkpoint_file = cp.get_last_checkpoint()
        start_step, start_epoch = _get_info_from_checkpoint4qat(checkpoint_file)
    elif cfg.TRAIN.WEIGHTS:
        checkpoint_file = cfg.TRAIN.WEIGHTS
        start_step, start_epoch = "1_bn_stabilization", 0

    teacher = None
    if str.lower(cfg.TRAIN.TEACHER) != "":
        teacher = setup_teacher()
        teacher.eval()
        restore_cfg()

    # Create a GradScaler for mixed precision training
    scaler = amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    if not checkpoint_file:
        logger.info("Start Calibration")
        calibration_loader = data_loader.construct_calibration_loader()
        calibrate_model(model, calibration_loader)
    model.apply(torch.quantization.disable_observer)
    ema.apply(torch.quantization.disable_observer)

    if cfg.QUANTIZATION.QAT.TRAIN_SHIFT_BIAS_QUANTIZATION:
        _enable_bias_quant(model)
        _enable_bias_quant(ema)

    _run_qat_newtork(
        model,
        ema,
        teacher,
        start_step,
        start_epoch,
        train_loader,
        test_loader,
        loss_fun,
        scaler,
        train_meter,
        test_meter,
        ema_meter,
        checkpoint_file,
    )


def get_optimizer_params(all_params, bn_stab: bool = False, scale_train: bool = True):
    w_param, bn_param, s_param = all_params
    scale_lr = (
        cfg.OPTIM.BASE_LR
        if cfg.QUANTIZATION.QAT.SCALE_LR == 0.0
        else cfg.QUANTIZATION.QAT.SCALE_LR
    )
    params = [
        {
            "params": w_param,
            "weight_decay": cfg.OPTIM.WEIGHT_DECAY,
            "lr": 0 if bn_stab else cfg.OPTIM.BASE_LR,
        },
        {
            "params": bn_param,
            "weight_decay": cfg.OPTIM.WEIGHT_DECAY,
            "lr": cfg.OPTIM.BASE_LR,
        },
        {
            "params": s_param,
            "weight_decay": 0,
            "lr": scale_lr if scale_train else 0,
        },
    ]

    if cfg.OPTIM.CLASS == "SGD":
        args = {
            "momentum": cfg.OPTIM.MOMENTUM,
            "dampening": cfg.OPTIM.DAMPENING,
            "nesterov": cfg.OPTIM.NESTEROV,
        }
    elif cfg.OPTIM.CLASS in ("Adam", "AdamW"):
        args = {}
    else:
        raise NotImplementedError

    return params, args


def _run_qat_newtork(
    model: Module,
    ema: Module,
    teacher: Module,
    start_step: str,
    start_epoch: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fun: Module,
    scaler: amp.GradScaler,
    train_meter: meters.TrainMeter,
    test_meter: meters.TrainMeter,
    ema_meter: meters.TrainMeter,
    checkpoint_file: str,
):
    # w_param, bn_param, s_param = _categorize_params(model)
    all_params = _categorize_params(model)
    step = start_step
    if step == "1_bn_stabilization":
        params, args = get_optimizer_params(all_params, True)
        stabilize_opt = optim.construct_optimizer(model, params, **args)
        if checkpoint_file:
            model, ema = _load_checkpoint(checkpoint_file, model, ema, stabilize_opt)
        logger.info("Start first BN stabilization (Epoch: {})".format(start_epoch + 1))
        _stabilize_bn(
            model,
            ema,
            teacher,
            start_epoch,
            train_loader,
            test_loader,
            loss_fun,
            stabilize_opt,
            scaler,
            train_meter,
            test_meter,
            ema_meter,
            prefix="1_bn_stabilization_",
        )
        start_epoch = 0
        step = "2_finetune"

    if step == "2_finetune":
        params, args = get_optimizer_params(all_params, scale_train=False)
        optimizer = optim.construct_optimizer(model, params, **args)
        if start_step == "2_finetune" and checkpoint_file:
            model, ema = _load_checkpoint(checkpoint_file, model, ema, optimizer)
        logger.info("Start finetuing (Epoch: {})".format(start_epoch + 1))
        _train_qat_newtork(
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
            bn_freeze=(cfg.QUANTIZATION.QAT.BN_TRAIN_EPOCH != -1),
            prefix="2_finetune_",
        )
        start_epoch = 0
        step = "3_bn_stabilization"

    if step == "3_bn_stabilization":
        params, args = get_optimizer_params(all_params, True)
        stabilize_opt = optim.construct_optimizer(model, params, **args)
        if start_step == "3_bn_stabilization" and checkpoint_file:
            model, ema = _load_checkpoint(checkpoint_file, model, ema, stabilize_opt)
        logger.info("Start second BN stabilization (Epoch: {})".format(start_epoch + 1))
        _stabilize_bn(
            model,
            ema,
            teacher,
            start_epoch,
            train_loader,
            test_loader,
            loss_fun,
            stabilize_opt,
            scaler,
            train_meter,
            test_meter,
            ema_meter,
            prefix="3_bn_stabilization_",
        )


@static_vars(prev_wlr=0.0, prev_slr=0.0)
def qat_lr_sched_func(optimizer, cur_epoch):
    ratio = optim.get_lr_fun()(cur_epoch)
    weight_lr = cfg.OPTIM.BASE_LR * ratio
    scale_lr = cfg.QUANTIZATION.QAT.SCALE_LR * ratio

    if cur_epoch == 0:
        qat_lr_sched_func.prev_wlr = cfg.OPTIM.BASE_LR
        qat_lr_sched_func.prev_slr = cfg.QUANTIZATION.QAT.SCALE_LR

    # Linear warmup
    if cur_epoch < cfg.OPTIM.WARMUP_EPOCHS:
        alpha = cur_epoch / cfg.OPTIM.WARMUP_EPOCHS
        warmup_factor = cfg.OPTIM.WARMUP_FACTOR * (1.0 - alpha) + alpha
        weight_lr *= warmup_factor
        scale_lr *= warmup_factor

    for param_group in optimizer.param_groups:
        if param_group["lr"] == qat_lr_sched_func.prev_wlr:
            param_group["lr"] = weight_lr
        elif param_group["lr"] == qat_lr_sched_func.prev_slr:
            param_group["lr"] = scale_lr
    qat_lr_sched_func.prev_wlr = weight_lr
    qat_lr_sched_func.prev_slr = scale_lr
    return weight_lr, scale_lr


def _train_qat_newtork(
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
        trainer.train_epoch(*params, cur_epoch, qat_lr_sched_func, teacher)

        if bn_freeze and cur_epoch >= cfg.QUANTIZATION.QAT.BN_TRAIN_EPOCH - 1:
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
