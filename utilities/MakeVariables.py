import numpy as np

nChannels = 24
inIceChannels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]


def getReferenceTraceIndices(vars, find_max=False):
    refIndex_PA = 0
    refIndex_inIce = 0
    refIndex_surface = 0

    if len(vars) != nChannels:
        raise ValueError("Need vars for all 24 channels...")
    else:
        vars_PA = np.array( vars[:4] )
        vars_inIce = np.array([])
        vars_surface = np.array([])

        for i in range(nChannels):
            if i in inIceChannels:
                vars_inIce = np.append(vars_inIce, vars[i])
            else:
                vars_surface = np.append(vars_surface, vars[i])

        if find_max:
            refIndex_PA = vars_PA.argmax()
            refIndex_inIce = vars_inIce.argmax()
            refIndex_surface = vars_surface.argmax()
        else:
            refIndex_PA = vars_PA.argmin()
            refIndex_inIce = vars_inIce.argmin()
            refIndex_surface = vars_surface.argmin()

    return refIndex_PA, refIndex_inIce, refIndex_surface
