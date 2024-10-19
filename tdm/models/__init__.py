import logging

logger = logging.getLogger("base")


def create_model(opt):
    model = opt["model"]

    if model == "denoising":
        from .denoising_model import DenoisingModel as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    m = M(opt)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m


def create_td_model(opt):
    model = opt["tensordec"]
    if model == "tucker":
        from .tensor_dec import TuckerModel as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    m = M(opt)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m
