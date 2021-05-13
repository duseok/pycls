"""Test a quantized classification model."""

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.quantization_tester as tester
from pycls.core.config import cfg

logger = logging.get_logger(__name__)


def main():
    config.load_cfg_fom_args("Test a quantized classification model.")
    config.assert_and_infer_cfg()
    if cfg.NUM_GPUS != 1:
        cfg.NUM_GPUS = 1
        logger.warning("When testing a quantized model, only one gpu can be used.")
    cfg.freeze()
    dist.multi_proc_run(num_proc=1, fun=tester.test_quantized_model)


if __name__ == "__main__":
    main()
