"""QAT (Quantization Aware Training)"""

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.quantization_trainer as trainer
from pycls.core.config import cfg

logger = logging.get_logger(__name__)


def main():
    config.load_cfg_fom_args("Train and test a network using QAT method.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.train_qat_network)


if __name__ == "__main__":
    main()
