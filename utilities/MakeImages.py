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


def imageHistogram(hist, traces, noiseLevels):
    nTraces = len(traces)
    bins = []

    for i_trace in range(nTraces):
        binValues = convertToBinValues( noiseLevels[i_trace], traces[i_trace], nTraces )
        bins.append(binValues)

    for i_trace in range(nTraces):
        for i_binX in range(nTraces):
            hist.SetBinContent( i_binX+1, i_trace+1, bins[i_trace][i_binX] )

    return bins


def imageBinsToVector(hist, vec):
    nh = hist.GetNbinsY()
    nw = hist.GetNbinsX()

    for j in range(nh):
        for i in range(nw):
            m = j * nw + i
            vec[m] = hist.GetBinContent( i+1, j+1 )
