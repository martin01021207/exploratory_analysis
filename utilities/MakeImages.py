import numpy as np
import ROOT


def convertToBinValues(noiseLevel, envelopeTrace, nParts):
    points = np.array([])
    envelope = np.array(envelopeTrace)
    traceParts = np.array_split(envelope, nParts)
    for i in range(nParts):
        point = np.amax(traceParts[i]) / noiseLevel
        points = np.append(points, point)

    return points


def imageHistogram(hist, envelopes, noiseLevels):
    nChannels = 24
    N = 48

    bins = []
    for i_channel in range(nChannels):
        binValues = convertToBinValues( noiseLevels[i_channel], envelopes[i_channel], N )
        bins.append(binValues)

    for i_channel in range(nChannels):
        for i_binX in range(N):
            hist.SetBinContent( i_binX+1, nChannels-i_channel, bins[i_channel][i_binX] )
            hist.SetBinContent( i_binX+1, nChannels+i_channel+1, bins[i_channel][i_binX] )
            
    return bins


def imageBinsToVector(hist, vec):
    nh = hist.GetNbinsY()
    nw = hist.GetNbinsX()

    for j in range(nh):
        for i in range(nw):
            m = j * nw + i
            vec[m] = hist.GetBinContent( i+1, j+1 )
