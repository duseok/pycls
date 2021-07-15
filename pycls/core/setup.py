from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import pycls.core.builders as builders
import pycls.core.checkpoint as cp
import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.net as net
import torch
from pycls.core.config import cfg
from pycls.core.io import pathmgr

if TYPE_CHECKING:
    from torch.nn import Module

logger = logging.get_logger(__name__)


def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure that the output dir exists
        pathmgr.mkdirs(cfg.OUT_DIR)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log torch, cuda, and cudnn versions
    version = [torch.__version__, torch.version.cuda, torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg)) if cfg.VERBOSE else ()
    logger.info(logging.dump_log_data(cfg, "cfg", None))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def model2cuda(model: Module):
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        ddp = torch.nn.parallel.DistributedDataParallel
        model = ddp(module=model, device_ids=[cur_device], output_device=cur_device)
    return model


def setup_model(to_cuda=True):
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.build_model()
    logger.info("Model:\n{}".format(model)) if cfg.VERBOSE else ()
    # Log model complexity
    logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    return model2cuda(model) if to_cuda else model


def setup_teacher():
    cfg.defrost()
    config.load_cfg_fom_args(description="Teacher model", cfg_file=cfg.TRAIN.TEACHER)
    config.assert_and_infer_cfg()
    cfg.freeze()
    teacher = setup_model(False)
    assert cfg.TRAIN.TEACHER_WEIGHTS != ""
    cp.load_checkpoint(cfg.TRAIN.TEACHER_WEIGHTS, teacher, None)
    logger.info(
        "Loaded initial teacher weights from: {}".format(cfg.TRAIN.TEACHER_WEIGHTS)
    )
    return model2cuda(teacher)


def restore_cfg():
    cfg.defrost()
    config.reset_cfg()
    config.load_cfg_fom_args("Restore a network configuration.")
    config.assert_and_infer_cfg()
    cfg.freeze()
