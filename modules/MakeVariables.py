import numpy as np

nChannels = 24
inIceChannels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]


def getReferenceTraceIndices(SNRs):
    refIndex_PA = 0
    refIndex_inIce = 0
    refIndex_surface = 0

    if len(SNRs) != nChannels:
        print("Need SNRs for all 24 channels...")
    else:
        SNRs_PA = np.array( SNRs[:4] )
        SNRs_inIce = np.array([])
        SNRs_surface = np.array([])

        for i in range(nChannels):
            if i in inIceChannels:
                SNRs_inIce = np.append(SNRs_inIce, SNRs[i])
            else:
                SNRs_surface = np.append(SNRs_surface, SNRs[i])

        refIndex_PA = SNRs_PA.argmax()
        refIndex_inIce = SNRs_inIce.argmax()
        refIndex_surface = SNRs_surface.argmax()

    return refIndex_PA, refIndex_inIce, refIndex_surface
