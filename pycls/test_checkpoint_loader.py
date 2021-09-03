# Weight loader

import os
import torch
import pycls.core.distributed as dist
from pycls.core.config import cfg
from pycls.core.io import pathmgr
from pycls.core.net import unwrap_model


def load_checkpoint(checkpoint_file):
    checkpoint_file = ""
    err_str = "Checkpoint '{}' not found"
    assert pathmgr.exists(checkpoint_file), err_str.foramt(checkpoint_file)
    with pathmgr.open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    for i in checkpoint:
        print(i, checkpoint[i])

    epoch = checkpoint["epoch"]
    test_err = checkpoint["test_err"]
    ema_err = checkpoint["ema_err"]
    model_state = checkpoint["model_state"]
    ema_state = checkpoint["ema_state"]
    optimizer_state = checkpoint["optimizer_state"]
    cfg = checkpoint["cfg"]
    step = checkpoint["step"]

    print(model_State['s15.b1.lin_proj_bn.running_var'])
    for key, val in model_state.items():
        print(key, val)


if __name__ == "__main__":
    load_checkpoint()
