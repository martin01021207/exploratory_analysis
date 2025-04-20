import numpy as np

nChannels = 24
inIceChannels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]
nChannels_inIce = len(inIceChannels)


def getReferenceTraceIndices(vars, find_max=False):
    if len(vars) not in {nChannels, nChannels_inIce}:
        raise ValueError("Need vars for all 24 channels or for all 15 in-ice channels.")

    inIceOnly = len(vars) == nChannels_inIce
    vars = np.array(vars)

    vars_PA = vars[:4]
    vars_inIce = vars if inIceOnly else vars[inIceChannels]
    vars_surface = [] if inIceOnly else np.delete(vars, inIceChannels)

    func = np.argmax if find_max else np.argmin

    refIndex_PA = func(vars_PA)
    refIndex_inIce = func(vars_inIce)
    refIndex_surface = func(vars_surface) if not inIceOnly else 0

    return refIndex_PA, refIndex_inIce, refIndex_surface
