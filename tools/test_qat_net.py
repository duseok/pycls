"""Test a QAT classification model."""

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.quantization_tester as tester
from pycls.core.config import cfg


def main():
    config.load_cfg_fom_args("Test a trained classification model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=tester.test_qat_network)


if __name__ == "__main__":
    main()
