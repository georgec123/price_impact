import logging
from logging import Logger


def get_logger(name) -> Logger:
    format = "[%(asctime)s] [%(levelname)s] [%(name)s - %(lineno)d] %(message)s"

    logger = logging.getLogger(name)
    console = logging.StreamHandler()
    formatter = logging.Formatter(format)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


class PoolExecutor:
    """
    Class to handle lambda functions for multiprocessing.
    Eg. create class with all other args, then pass instance instead of lambda function

    .apply(lambda x: fn(x, y)  )
    becomes
    pe_fn = PoolExecutor(fn, y)
    .apply(pe_fn)
    """

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.fn(x, *self.args, **self.kwargs)
