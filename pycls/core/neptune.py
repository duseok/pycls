from __future__ import annotations

from pycls.core.config import cfg
from yacs.config import CfgNode

import neptune.new as neptune

_C = CfgNode()
_C.PROJECT = ""
_C.TOKEN = ""
_C.TAGS = []


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class NeptuneLogger(metaclass=SingletonMeta):
    def __init__(self):
        self.__load_cfg()
        self.nt = neptune.init(
            project=_C.PROJECT if cfg.NEPTUNE_PROJECT == "" else cfg.NEPTUNE_PROJECT,
            api_token=_C.TOKEN,
            tags=_C.TAGS,
            run=None if cfg.NEPTUNE_RESUME == "" else cfg.NEPTUNE_RESUME,
        )
        self.nt["Config"] = cfg

    @staticmethod
    def __load_cfg():
        _C.merge_from_file(cfg.NEPTUNE_CONFIG)
        _C.TAGS += cfg.NEPTUNE_TAGS
        _C.freeze()

    def log(self, name: str, value: int or float):
        self.nt[name].log(value)

    def sync(self):
        self.nt.sync(wait=False)
