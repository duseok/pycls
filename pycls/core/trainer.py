#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

from copy import deepcopy

import pycls.core.benchmark as benchmark
import pycls.core.builders as builders
import pycls.core.checkpoint as cp
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.meters as meters
import pycls.core.net as net
import pycls.core.optimizer as optim
import pycls.datasets.loader as data_loader
import torch
import torch.cuda.amp as amp
from pycls.core.config import cfg
from pycls.core.setup import restore_cfg, setup_env, setup_model, setup_teacher

logger = logging.get_logger(__name__)


def get_kd_loss(output, output_t):
    from torch.nn.functional import log_softmax, softmax

    return -1 * torch.mean(
        torch.sum(softmax(output_t, dim=1) * log_softmax(output, dim=1), dim=1)
    )


def default_lr_sched_func(optimizer, cur_epoch):
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    return lr, None


def teacher_student_loss(teacher, inputs, preds):
    if not teacher:
        return 0.0

    with torch.no_grad():
        preds_t = teacher(inputs)
        preds_t_scaled = temperature_scale(preds_t, cfg.TRAIN.TEMPERATURE)
    loss = get_kd_loss(preds, preds_t_scaled)
    return loss


def quant_loss(model: torch.nn.Module):
    from pycls.quantization.shift_fake_quantizer import ShiftFakeQuantize

    if not cfg.QUANTIZATION.QAT.ENABLE_QUANTIZATION_LOSS:
        return 0.0

    loss = 0.0
    for _, m in model.named_modules():
        if isinstance(m, ShiftFakeQuantize) and m.quant_loss is not None:
            loss += m.quant_loss
    return loss


def temperature_scale(preds, temperature):
    return preds if temperature == 1.0 else torch.div(preds, temperature)


def train_epoch(
    loader,
    model,
    ema,
    loss_fun,
    optimizer,
    scaler,
    meter,
    cur_epoch,
    lr_sched_func=default_lr_sched_func,
    teacher=None,
):
    """Performs one epoch of training."""
    # Shuffle the data
    data_loader.shuffle(loader, cur_epoch)
    # Update the learning rate
    weight_lr, scale_lr = lr_sched_func(optimizer, cur_epoch)
    # Enable training mode
    model.train()
    ema.train()
    meter.reset()
    meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Convert labels to smoothed one-hot vector
        labels_one_hot = net.smooth_one_hot_labels(labels)
        # Apply mixup to the batch (no effect if mixup alpha is 0)
        inputs, labels_one_hot, labels = net.mixup(inputs, labels_one_hot)
        # Perform the forward pass and compute the loss
        with amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            preds = model(inputs)
            preds_scaled = temperature_scale(preds, cfg.TRAIN.TEMPERATURE)
            loss = (
                loss_fun(preds if teacher else preds_scaled, labels_one_hot)
                + teacher_student_loss(teacher, inputs, preds_scaled)
                + (quant_loss(model) * cfg.QUANTIZATION.QAT.QUANTIZATION_LOSS_ALPHA)
            )
        # Perform the backward pass and update the parameters
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Update ema weights
        net.update_model_ema(model, ema, cur_epoch, cur_iter)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        meter.iter_toc()
        # Update and log stats
        mb_size = inputs.size(0) * cfg.NUM_GPUS
        meter.update_stats(top1_err, top5_err, loss, weight_lr, mb_size, scale_lr)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)


@torch.no_grad()
def test_epoch(loader, model, meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    meter.reset()
    meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        meter.iter_toc()
        # Update and log stats
        meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)


def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model, ema, loss_fun, and optimizer
    model = setup_model()
    ema = deepcopy(model)
    loss_fun = builders.build_loss_fun().cuda()
    optimizer = optim.construct_optimizer(model)
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and cp.has_checkpoint():
        file = cp.get_last_checkpoint()
        epoch = cp.load_checkpoint(file, model, ema, optimizer)[0]
        logger.info("Loaded checkpoint from: {}".format(file))
        start_epoch = epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        cp.load_checkpoint(cfg.TRAIN.WEIGHTS, model, ema)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = data_loader.construct_train_loader()
    test_loader = data_loader.construct_test_loader()
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    ema_meter = meters.TestMeter(len(test_loader), "test_ema")

    teacher = None
    if str.lower(cfg.TRAIN.TEACHER) != "":
        teacher = setup_teacher()
        teacher.eval()
        restore_cfg()

    # Create a GradScaler for mixed precision training
    scaler = amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    # Compute model and loader timings
    if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
        benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        params = (train_loader, model, ema, loss_fun, optimizer, scaler, train_meter)
        train_epoch(*params, cur_epoch, default_lr_sched_func, teacher)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
            net.compute_precise_bn_stats(ema, train_loader)
        # Evaluate the model
        test_epoch(test_loader, model, test_meter, cur_epoch)
        test_epoch(test_loader, ema, ema_meter, cur_epoch)
        test_err = test_meter.get_epoch_stats(cur_epoch)["top1_err"]
        ema_err = ema_meter.get_epoch_stats(cur_epoch)["top1_err"]
        # Save a checkpoint
        file = cp.save_checkpoint(model, ema, optimizer, cur_epoch, test_err, ema_err)
        logger.info("Wrote checkpoint to: {}".format(file))


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model()
    # Load model weights
    cp.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = data_loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)


def time_model():
    """Times model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Compute model and loader timings
    benchmark.compute_time_model(model, loss_fun)


def time_model_and_loader():
    """Times model and data loader."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Create data loaders
    train_loader = data_loader.construct_train_loader()
    test_loader = data_loader.construct_test_loader()
    # Compute model and loader timings
    benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
