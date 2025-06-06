import argparse
import logging
import numpy as np
from array import array

import NuRadioReco.modules.RNO_G.stationHitFilter
from NuRadioReco.utilities import units, trace_utilities, logging as nulogging

import ROOT
from ROOT import TFile, TTree, TGraph


logger = logging.getLogger("NuRadioReco.example.RNOG.apply_hit_filter")
logger.setLevel(nulogging.LOGGING_STATUS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_in", type=str, help="Input directory")
    parser.add_argument("dir_out", type=str, help="Output directory")
    parser.add_argument("station_number", type=int, help="Station number")
    parser.add_argument("--run", type=int, help="Run number")
    parser.add_argument('--json_select', type=str, default=None, help="JSON file of events to select")
    parser.add_argument("--isExcluded", action="store_true", help="Exclude events in JSON")
    parser.add_argument("--uproot", action="store_true", help="Using uproot backend")
    parser.add_argument("--isSim", action="store_true", help="Simulation input")
    parser.add_argument("--CR", action="store_true", help="CR simulation")
    parser.add_argument("--sim_E", type=str, help="Simulation energy")
    args = parser.parse_args()

    nulogging.set_general_log_level(logging.ERROR)

    dir_in = args.dir_in
    if not dir_in.endswith("/"):
        dir_in += "/"

    dir_out = args.dir_out
    if not dir_out.endswith("/"):
        dir_out += "/"

    stationNumber = args.station_number
    runNumber = args.run

    highTrigRuns = np.loadtxt(f"/work/hep/martinliu/realData/triggerRates/highTrigRuns_s{stationNumber}.txt", dtype=int)
    if runNumber in highTrigRuns:
        quit()

    json_select = args.json_select
    isExcluded = args.isExcluded

    if args.uproot:
        backend = 'uproot'
    else:
        backend = 'pyroot'

    isSim = args.isSim
    isCR = args.CR
    sim_E = args.sim_E

    if sim_E:
        if not sim_E.endswith("eV"):
            sim_E += "eV"
        sim_E_number = float( sim_E.replace('eV','') )

    isSelectingEvents = False
    if json_select:
        isSelectingEvents = True
        import json
        with open(json_select, 'r') as json_selectEvents:
            selection = json.loads(json_selectEvents.read())
        if str(runNumber) in selection:
            eventList = selection[str(runNumber)]
        else:
            #eventList = []
            #if not isExcluded:
            print(f"Run {runNumber} is not in the list, exit now.")
            quit()

    fileList = []

    if isSim:
        if isCR:
            import NuRadioReco.modules.io.NuRadioRecoio as NuRadioRecoio
        import os
        import NuRadioReco.modules.io.eventReader as readSimData
        treename = "events_sim"
        filename_out = f"filtered_sim_s{stationNumber}_{sim_E}.root"
        for data in os.listdir(dir_in):
            if data.endswith(".nur"):
                fileList.append(dir_in+str(data))
    else:
        import NuRadioReco.modules.RNO_G.dataProviderRNOG
        import NuRadioReco.modules.channelResampler
        import NuRadioReco.modules.channelBandPassFilter
        import NuRadioReco.modules.channelCWNotchFilter
        import NuRadioReco.detector.RNO_G.rnog_detector
        import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
        treename = "events"
        filename_out = f"filtered_s{stationNumber}_r{runNumber}.root"
        data = f"station{stationNumber}/run{runNumber}/"
        fileList.append(dir_in+data)

    nChannels = 24
    graph_vector = ROOT.std.vector["TGraph"](nChannels)
    station_number = array('i', [0])
    run_number = array('i', [0])
    event_number = array('i', [0])
    sim_energy = array('f', [0.])
    trigger_time_difference = array('f', [0.])

    tree_out = ROOT.TTree(treename, treename)

    file_out = TFile(dir_out+filename_out, "RECREATE")

    tree_out.Branch("waveform_graphs", "std::vector<TGraph>", graph_vector)
    tree_out.Branch("station_number", station_number, 'station_number/I')
    tree_out.Branch("run_number", run_number, 'run_number/I')
    tree_out.Branch("event_number", event_number, 'event_number/I')
    tree_out.Branch("sim_energy", sim_energy, 'sim_energy/F')
    tree_out.Branch("trigger_time_difference", trigger_time_difference, 'trigger_time_difference/F')
    tree_out.SetDirectory(file_out)
    tree_out.SetMaxTreeSize(1000000000000)

    # Initialize Hit Filter
    stationHitFilter = NuRadioReco.modules.RNO_G.stationHitFilter.stationHitFilter()
    stationHitFilter.begin()

    runNumber_sim = 0
    nEvents_FT = 0
    nEvents_RADIANT = 0
    nEvents_badSim = 0
    nEvents_total = 0
    nEvents_passedHF = 0
    for file in fileList:
        if isSim:
            det = None
            if not isCR:
                runNumber_sim = int(file.split("/")[-1].split(".nur")[0].split("_")[2])
            else:
                nur_reader = NuRadioRecoio.NuRadioRecoio(file)
                event_headers = nur_reader.get_event_ids()
                cosine = float(file.split("/")[-1].split(".nur")[0].split("-")[1].split("_")[1])
                theta = int(round(np.arccos(cosine) * (180 / np.pi), 0))
                phi = int(float(file.split("/")[-1].split(".nur")[0].split("-")[2].split("_")[1]))
            eventID_sim = 0
            reader = readSimData.eventReader()
            reader.begin(file)
        else:
            det = NuRadioReco.detector.RNO_G.rnog_detector.Detector(detector_file=None)
            reader = NuRadioReco.modules.RNO_G.dataProviderRNOG.dataProviderRNOG()
            reader.begin(files=file, det=det, reader_kwargs={'mattak_kwargs': {'backend': backend}})
            channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
            channelResampler.begin()
            channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
            channelBandPassFilter.begin()
            channelCWNotchFilter = NuRadioReco.modules.channelCWNotchFilter.channelCWNotchFilter()
            channelCWNotchFilter.begin()
            hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
            hardwareResponseIncorporator.begin()
            info = reader.reader.get_events_information(keys=["station", "run", "eventNumber", "triggerType", "triggerTime"])

        for i_event, event in enumerate(reader.run()):
            station = event.get_station()
            station_id = station.get_id()

            isBadSimEvent = False

            station_number[0] = station_id
            if isSim:
                run_number[0] = runNumber_sim
                event_number[0] = eventID_sim
                sim_energy[0] = sim_E_number
                trigger_time_difference[0] = 0.
                eventID_sim += 1
            else:
                run_number[0] = info[i_event].get('run')
                event_number[0] = info[i_event].get('eventNumber')
                sim_energy[0] = 0.
                trigger_time_difference[0] = info[i_event].get('triggerTime') - info[0].get('triggerTime')
                trigger_type = info[i_event].get('triggerType')
                if trigger_type == "FORCE":
                    nEvents_FT += 1
                    continue
                elif "RADIANT" in trigger_type:
                    nEvents_RADIANT += 1
                    continue

            if isSelectingEvents:
                if isExcluded:
                    if event_number[0] in eventList:
                        continue
                else:
                    if event_number[0] not in eventList:
                        continue

            if not isSim:
                det.update(station.get_station_time())
                channelResampler.run(event, station, det, sampling_rate=5 * units.GHz)
                channelBandPassFilter.run(
                    event, station, det,
                    passband=[0.1 * units.GHz, 0.6 * units.GHz],
                    filter_type='butter', order=10)
                hardwareResponseIncorporator.run(event, station, det, sim_to_data=False, mode='phase_only')
                channelCWNotchFilter.run(event, station, det)

            traces = []
            times = []
            channels = np.array([])
            for i_channel, channel in enumerate(station.iter_channels()):
                channel_id = channel.get_id()
                y = np.array(channel.get_trace()) / units.mV
                x = np.array(channel.get_times())

                if isCR:
                    y = np.roll(y, 800)

                if isSim:
                    isBadTrace, _, _ = trace_utilities.is_NAN_or_INF(y)
                    if isBadTrace:
                        isBadSimEvent = True
                        break

                traces.append(y)
                times.append(x)
                channels = np.append(channels, channel_id)

            traces = np.array(traces)
            times = np.array(times)
            sorted_indices = channels.argsort()
            sorted_channels = np.sort(channels)
            traces = traces[sorted_indices]
            times = times[sorted_indices]

            for i_channel in sorted_channels:
                i_channel = int(i_channel)
                graph_vector[i_channel] = TGraph(len(times[i_channel]), times[i_channel], traces[i_channel])
                graph_vector[i_channel].GetXaxis().SetTitle("time [ns]")
                graph_vector[i_channel].GetYaxis().SetTitle("amplitude [mV]")
                if isCR:
                    graphTitle = f"({sim_energy[0]},{theta},{phi}): Evt{event_number[0]}, Ch{i_channel}"
                else:
                    graphTitle = f"S{station_number[0]}, R{run_number[0]}, Evt{event_number[0]}, Ch{i_channel}"
                graph_vector[i_channel].SetTitle(graphTitle)

            if isBadSimEvent:
                nEvents_badSim += 1
                continue

            # Hit Filter
            is_passed_HF = stationHitFilter.run(event, station, det)

            nEvents_total += 1
            if is_passed_HF:
                nEvents_passedHF += 1
                tree_out.Fill()

        reader.end()
        if isCR:
            nur_reader.close_files()
            runNumber_sim += 1

    file_out.cd()
    tree_out.Write()
    file_out.Close()

    if isSim:
        print(f"Station {stationNumber}  Energy {sim_E}")
        print(f"Number of BAD sim events: {nEvents_badSim}")
    else:
        print(f"Station {stationNumber}  Run {runNumber}")
        print(f"Number of forced trigger events: {nEvents_FT}")
        print(f"Number of RADIANT trigger events: {nEvents_RADIANT}")
    print(f"Number of total RF events: {nEvents_total}")
    print(f"Number of RF events passed the hitFilter: {nEvents_passedHF}")
