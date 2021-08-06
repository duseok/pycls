import sys


def str_to_class(mod_name, cls_name):
    return getattr(sys.modules[mod_name], cls_name)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate
