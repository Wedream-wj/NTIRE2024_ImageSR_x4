import logging
logger = logging.getLogger('base')


def RepRLFN(checkpoint, deploy=True):
    from .RLFN import get_RLFN
    model = get_RLFN(checkpoint=checkpoint, deploy=deploy)
    return model