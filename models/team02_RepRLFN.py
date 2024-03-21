from .RepRLFN import RepRLFN as Rep


def RepRLFN(checkpoint, deploy):
    return Rep(checkpoint=checkpoint, deploy=deploy)
