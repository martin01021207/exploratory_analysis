import numpy as np
import WaveformAnalysisTools as WfAT


class HitFilter:
    def __init__(self, completeTimeSequenceCheck = False, completeHitCheck = False, timeWindow = 10.0, multiHitThresholds = False, thresholdMultipliers = [6.5, 6.0, 5.5, 5.0, 4.5]):
        self._passedTimeChecker = False
        self._passedHitChecker = False
        self._completeTimeSequenceCheck = completeTimeSequenceCheck
        self._completeHitCheck = completeHitCheck
        self.dT = timeWindow
        self._multiHitThresholds = multiHitThresholds
        self._thresholdMultipliers = thresholdMultipliers
        if not self._multiHitThresholds:
            self._thresholdMultipliers = [ self._thresholdMultipliers[0] ]
        self._nThresholds = len(self._thresholdMultipliers)
        self._inIceChannels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]
        self._inIceChannelGroups = ([0, 1, 2, 3], [9, 10], [23, 22], [8, 4], [5], [6], [7], [11], [21])
        self._channelPairsInPA = ([0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3])

        self._nInIceChannels = len(self._inIceChannels)
        self._inTimeSequence = []
        for i_group, group in enumerate(self._inIceChannelGroups):
            if i_group == 0:
                self._inTimeSequence.append([])
                for i_channelPair in range(len(self._channelPairsInPA)):
                    self._inTimeSequence[i_group].append(None)
            elif len(group) > 1:
                self._inTimeSequence.append([])
                for i_channelPair in range(len(group)-1):
                    self._inTimeSequence[i_group].append(None)

        self._rolledTraces = []
        self._corrX = []
        self._corrY = []
        self._envelopeTimes = [ [] for i in range(self._nInIceChannels) ]
        self._traces = [ [] for i in range(self._nInIceChannels) ]
        self._envelopes = [ [] for i in range(self._nInIceChannels) ]
        self._envelopeMaxTimeIndex = np.array( [None] * self._nInIceChannels )
        self._envelopeMaxTime = np.array( [None] * self._nInIceChannels )
        self._noiseRMS = np.array( [None] * self._nInIceChannels )
        self._hitThresholds = [ [] for i in range(self._nInIceChannels) ]
        self._overHitThreshold = [ [] for i in range(self._nInIceChannels) ]

        for i_channel in range(self._nInIceChannels):
            for i in range(self._nThresholds):
                self._hitThresholds[i_channel].append(None)
                self._overHitThreshold[i_channel].append(None)


    ### Channel Mapping ###
    # Hit Filter works only on all 15 in-ice channels,
    # and we map channels 21, 22, 23 to indices 12, 13, 14 in arrays
    # (Skip surface channels 12 ~ 20)
    def channelMapping(self, channel_in):
        if channel_in >= 21:
            channel_out = channel_in - 9
        else:
            channel_out = channel_in
        return channel_out


    #####################
    ###### Getters ######
    #####################

    def getChannelInGroups(self):
        return self._inIceChannelGroups

    def getEnvelopeTimes(self):
        return np.array(self._envelopeTimes)

    def getEnvelopes(self):
        return np.array(self._envelopes)

    def getEnvelopeMaxTimeIndex(self):
        return self._envelopeMaxTimeIndex

    def getEnvelopeMaxTime(self):
        return self._envelopeMaxTime

    def getRMS_allInIceChannels(self):
        return self._noiseRMS

    def getThresholdMultipliers(self):
        return self._thresholdMultipliers

    def getHitThresholds(self):
        return np.array(self._hitThresholds)

    def getTraces(self):
        return np.array(self._traces)

    def getRolledTraces(self):
        return np.array(self._rolledTraces)

    def getCorrX(self):
        return np.array(self._corrX)

    def getCorrY(self):
        return np.array(self._corrY)

    def isInTimeSequence(self):
        return self._inTimeSequence

    def isOverHitThreshold(self):
        return self._overHitThreshold

    def isPassedTimeChecker(self):
        return self._passedTimeChecker

    def isPassedHitChecker(self):
        return self._passedHitChecker


    ### Setting Up For The Hit Filter ###
    # Calculate noise RMS, get envelope, and find the time when maximum happens
    def setup(self, eventTraces, eventTimes, RMS):
        for group in self._inIceChannelGroups:
            for channel in group:
                mappedChannel = self.channelMapping(channel)
                if RMS.size != 0:
                    self._noiseRMS[mappedChannel] = RMS[channel]
                else:
                    self._noiseRMS[mappedChannel] = WfAT.getNoiseRMS(eventTraces[channel])
                for i in range(self._nThresholds):
                    self._hitThresholds[mappedChannel][i] = self._noiseRMS[mappedChannel] * self.getThresholdMultipliers()[i]
                self._traces[mappedChannel] = eventTraces[channel]
                self._envelopes[mappedChannel], self._envelopeTimes[mappedChannel] = WfAT.getHilbertEnvelope_waveform(eventTraces[channel], eventTimes[channel])
                self._envelopeMaxTimeIndex[mappedChannel] = np.array(self._envelopes[mappedChannel]).argmax()
                self._envelopeMaxTime[mappedChannel] = self._envelopeTimes[mappedChannel][self._envelopeMaxTimeIndex[mappedChannel]]


    ### Time Sequence Checker ###
    # Find at least 2 coincident pairs in Group 1 (PA),
    # if only 1 pair in PA, then find the other pair in other groups.
    def timeSequenceChecker(self):
        self._passedTimeChecker = False
        envelopeMaxTime = self.getEnvelopeMaxTime()
        isCoincidentInPA = [False, False, False]

        for i_group, group in enumerate(self._inIceChannelGroups):
            if i_group == 0:
                for i_channelPair, channelPair in enumerate(self._channelPairsInPA):
                    if abs(envelopeMaxTime[channelPair[1]] - envelopeMaxTime[channelPair[0]]) <= self.dT * abs(channelPair[1] - channelPair[0]):
                        self._inTimeSequence[i_group][i_channelPair] = True
                    else:
                        self._inTimeSequence[i_group][i_channelPair] = False
                    if channelPair[0] == 0 and self._inTimeSequence[i_group][i_channelPair]:
                        isCoincidentInPA[0] = True
                    elif channelPair[0] == 1 and self._inTimeSequence[i_group][i_channelPair]:
                        isCoincidentInPA[1] = True
                    elif channelPair[0] == 2 and self._inTimeSequence[i_group][i_channelPair]:
                        isCoincidentInPA[2] = True
                if sum(isCoincidentInPA) >= 2:
                    self._passedTimeChecker = True
                    if not self._completeTimeSequenceCheck:
                        break
            elif len(group) > 1:
                if abs(envelopeMaxTime[self.channelMapping(group[0])] - envelopeMaxTime[self.channelMapping(group[1])]) <= self.dT:
                    self._inTimeSequence[i_group][0] = True
                else:
                    self._inTimeSequence[i_group][0] = False

                if self._inTimeSequence[i_group][0] and sum(isCoincidentInPA) >= 1:
                    self._passedTimeChecker = True
                    if not self._completeTimeSequenceCheck:
                        break
            else:
                break

        return self._passedTimeChecker


    ### Hit Checker ###
    def hitChecker(self):
        self._passedHitChecker = False
        envelopes = self.getEnvelopes()
        hitThresholds = self.getHitThresholds()
        aboveThresholdCounts = np.zeros(self._nThresholds)

        for i_channel in range(self._nInIceChannels):
            for i in range(self._nThresholds):
                if np.amax(envelopes[i_channel]) > hitThresholds[i_channel][i]:
                    self._overHitThreshold[i_channel][i] = True
                    aboveThresholdCounts[i] += 1
                else:
                    self._overHitThreshold[i_channel][i] = False

        for i in range(self._nThresholds):
            if aboveThresholdCounts[i] > i:
                self._passedHitChecker = True

        return self._passedHitChecker


    ### See If Event Survived The Hit Filter? ###
    def passedHitFilter(self, eventTraces, eventTimes, RMS=np.array([])):
        passedFilter = False

        self.setup(eventTraces, eventTimes, RMS)
        self.timeSequenceChecker()

        if not self._completeHitCheck:
            if self.isPassedTimeChecker():
                passedFilter = True
                return passedFilter
            else:
                self.hitChecker()
                if self.isPassedHitChecker():
                    passedFilter = True
        else:
            self.hitChecker()
            if self.isPassedTimeChecker() or self.isPassedHitChecker():
                passedFilter = True

        return passedFilter
