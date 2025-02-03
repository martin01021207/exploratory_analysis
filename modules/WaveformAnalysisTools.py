import numpy as np


### Array Rolling ###
# Use known shift in samples to roll an array.
def roll_array(inputArray, shiftInSamples):
    rolledArray = np.roll( inputArray, shiftInSamples )

    return rolledArray


### Check Waveform Quality ###
# See if there's NAN or INF in the waveform trace
def isBadWaveform(trace):
    badWaveformFound = False

    trace = np.array(trace)

    nPoints_nan = len( np.argwhere(np.isnan(trace)) )
    nPoints_inf = len( np.argwhere(np.isinf(trace)) )

    if nPoints_nan or nPoints_inf:
        badWaveformFound = True

    return badWaveformFound


#########################################################################################
###  PART I: Some Basic Functions For Waveform Analysis:                              ###
###  1. Hilbert Transform                                                             ###
###  2. Noise RMS Calculation                                                         ###
###  3. Signal To Noise Ratio                                                         ###
###  4. Cross Correlation Calculation                                                 ###
#########################################################################################
from scipy.signal import hilbert, correlate, correlation_lags, argrelextrema


### Hilbert Transform (Trace Only) ###
# This function applies the Hilbert Transform on a waveform trace, and return the envelope trace.
def getHilbertEnvelope_trace(inputTrace):
    # Get the Hilbert envelope of the waveform trace
    envelope = np.abs(hilbert(inputTrace))

    return envelope


### Hilbert Transform (Trace & Times) ###
# This function applies the Hilbert Transform on a waveform, and return the envelope.
def getHilbertEnvelope_waveform(inputTrace, inputTimes):
    # Get the Hilbert envelope of the waveform trace
    envelope = np.abs(hilbert(inputTrace))
    times = inputTimes

    return envelope, times


### Noise RMS ###
# This function calculates the noise RMS for a trace (envelope or waveform).
# It breaks down the trace into 4 parts and calculates the RMS of each part,
# then it will take the average of the two smallest RMS values to be the noise RMS of the trace.
def getNoiseRMS(inputTrace):
    nParts = 4
    traceInParts = np.array_split(inputTrace, nParts)
    partRMS = [ np.sqrt(np.mean(part**2)) for part in traceInParts ]
    partRMS.sort()
    noiseRMS = np.mean(partRMS[:2])

    return noiseRMS


### Signal To Noise Ratio ###
# Input calculated noise RMS to find the SNR of a trace.
def getSNR(inputTrace, RMS):
    upper_peak_idx = argrelextrema(inputTrace, np.greater_equal, order = 1)[0]
    lower_peak_idx = argrelextrema(inputTrace, np.less_equal, order = 1)[0]
    peak_idx = np.unique(np.concatenate((upper_peak_idx, lower_peak_idx)))

    peak = inputTrace[peak_idx]
    P2P = np.abs(np.diff(peak))
    P2P = np.nanmax(P2P)
    SNR = P2P / (2 * RMS)

    del upper_peak_idx, lower_peak_idx, peak_idx, peak

    return SNR


### Cross Correlation ###
# This function finds the cross correlation between two traces (envelopes or waveforms).
# It returns two arrays: cross correlation & lag in samples
def getCrossCorrelation(trace_1, trace_2):
    trace1 = np.array(trace_1)
    trace2 = np.array(trace_2)

    nSamples_trace1 = len(trace1)
    nSamples_trace2 = len(trace2)

    trace1 /= np.amax(trace1)
    trace1 /= np.std(trace1)
    trace1 -= np.mean(trace1)
    trace2 /= np.amax(trace2)
    trace2 /= np.std(trace2)
    trace2 -= np.mean(trace2)

    amplitudeFactor = 1.0 / float(nSamples_trace1)

    if nSamples_trace1 != nSamples_trace2:
        print("The input two arrays have different numbers of samples, quit cross correlation calculating...")
    else:
        # Get the cross correlation between two traces
        corr = amplitudeFactor * correlate(trace1, trace2)

        # Get the lags between two traces
        lags = correlation_lags(nSamples_trace1, nSamples_trace2)

        del trace1, trace2

        return np.array(corr), np.array(lags)


###############################################
### PART II: Other Waveform Analysis Tools  ###
###############################################
from scipy.stats import kurtosis, entropy


### Get Reference Trace's Index For A Set Of Traces ###
# Choose the reference trace's index based on the highest SNR
# Three groups (sets): PA, All In-Ice Channels, Surface Channels
def getReferenceTraceIndices(SNRs):
    inIceChannels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]

    nChannels = 24

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


### Coherently-Summed Waveform (CSW) Trace ###
# Find the cross correlation of traces between each channel with an assigned reference trace (default index 0),
# use the cross correlation time lag to roll each trace,
# then add up all traces,
# we get one coherently-summed waveform trace.
def getCSW_trace(array2D_trace, reference = 0, envelopesCorr = False):
    nChannels = len(array2D_trace)

    rolledTraces = [ [] for i in range(nChannels) ]
    corrX = [ [] for i in range(nChannels) ]
    corrY = [ [] for i in range(nChannels) ]

    trace_0 = np.array(array2D_trace[reference])
    if envelopesCorr:
        trace_0 = getHilbertEnvelope_trace(trace_0)

    CSW_trace = np.zeros(len(trace_0))

    for i in range(nChannels):
        trace_i = np.array(array2D_trace[i])
        if envelopesCorr:
            trace_i = getHilbertEnvelope_trace(trace_i)

        corrY[i], corrX[i] = getCrossCorrelation(trace_0, trace_i)
        corrMaxIdx = corrY[i].argmax()
        rolledTraces[i] = roll_array(trace_i, corrX[i][corrMaxIdx])
        CSW_trace += rolledTraces[i]

    return CSW_trace, rolledTraces, corrY, corrX


### Get Cross Correlation Maximum ###
def getCorrMax(array2D_corrY, reference = 0):
    nChannels = len(array2D_corrY)

    corrSum = np.zeros( len(array2D_corrY[reference]) )

    for i in range(nChannels):
        if i == reference:
            continue
        else:
            corrSum += array2D_corrY[i]

    corrMax = np.amax(corrSum)

    return corrMax


### Impulsivity ###
def getImpulsivity(CSW_trace):
    # Get the envelope trace of the CSW
    envelope = getHilbertEnvelope_trace(CSW_trace)
    # Find the index of the envelope peak
    peakAtIndex = envelope.argmax()
    # Indices in an array: [0, 1, 2, ...]
    indices = np.linspace(0, len(envelope)-1, len(envelope))
    # How many samples away from the peak
    nSamplesFromPeak = np.abs(indices - peakAtIndex)
    # Sort the envelope trace based on how close the point is from the peak
    envelope_sorted = [y for delta_x, y in sorted(zip(nSamplesFromPeak, envelope))]
    # Cumulative Distribution Function (CDF)
    CDF = np.cumsum(envelope_sorted)
    # Normalization
    CDF = CDF / CDF[-1]
    # Impulsivity = 2 * average - 1
    impulsivity = 2.0 * np.mean(CDF) - 1.0
    # Make sure the value is not less than 0
    if(impulsivity < 0):
        impulsivity = 0.0

    return impulsivity, CDF, indices


### Kurtosis ###
def getKurtosis(inputTrace):
    kurt = kurtosis(inputTrace)

    return kurt


### Shannon Entropy ###
def getEntropy(inputTrace, nHistBins = 50):
    # Step 1: Discretize the signal into bins
    # If density = True, the result is the value of the probability density function at the bin,
    # normalized such that the integral over the range is 1.
    hist, binEdges = np.histogram(inputTrace, bins = nHistBins, density = True)

    # Step 2: Calculate the probability distribution (normalized)
    probabilities = hist / np.sum(hist)

    # Step 3: Calculate Shannon Entropy
    # Using base = 2 for entropy in bits
    signalEntropy = entropy(probabilities, base = 2)

    return signalEntropy
