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

    json_select = args.json_select
    isExcluded = args.isExcluded

    if args.uproot:
        backend = 'uproot'
    else:
        backend = 'pyroot'

    isSim = args.isSim
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
            eventList = []
            if not isExcluded:
                print(f"Run {runNumber} is not in the list, exit now.")
                quit()

    fileList = []

    if isSim:
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

    tree_out = ROOT.TTree(treename, treename)

    file_out = TFile(dir_out+filename_out, "RECREATE")

    tree_out.Branch("waveform_graphs", "std::vector<TGraph>", graph_vector)
    tree_out.Branch("station_number", station_number, 'station_number/I')
    tree_out.Branch("run_number", run_number, 'run_number/I')
    tree_out.Branch("event_number", event_number, 'event_number/I')
    tree_out.Branch("sim_energy", sim_energy, 'sim_energy/F')
    tree_out.SetDirectory(file_out)

    # Initialize Hit Filter
    stationHitFilter = NuRadioReco.modules.RNO_G.stationHitFilter.stationHitFilter()
    stationHitFilter.begin()

    nEvents_badSim = 0
    nEvents_FT = 0
    nEvents_total = 0
    nEvents_passedHF = 0
    for file in fileList:
        if isSim:
            det = None
            runNumber = int(file.split("/")[-1].split(".nur")[0].split("_")[2])
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
            info = reader.reader.get_events_information(keys=["station", "run", "eventNumber", "triggerType"])

        for i_event, event in enumerate(reader.run()):
            station = event.get_station()
            station_id = station.get_id()

            isBadSimEvent = False

            station_number[0] = station_id
            if isSim:
                run_number[0] = runNumber
                event_number[0] = eventID_sim
                sim_energy[0] = sim_E_number
                eventID_sim += 1
            else:
                run_number[0] = info[i_event].get('run')
                event_number[0] = info[i_event].get('eventNumber')
                sim_energy[0] = 0.
                if info[i_event].get('triggerType') == "FORCE":
                    nEvents_FT += 1
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

            for i_channel, channel in enumerate(station.iter_channels()):
                channel_id = channel.get_id()
                y = np.array(channel.get_trace())
                x = np.array(channel.get_times())

                if isSim:
                    isBadTrace, _, _ = trace_utilities.is_NAN_or_INF(y)
                    if isBadTrace:
                        isBadSimEvent = True
                        break

                graph_vector[i_channel] = TGraph(len(x), x, y)
                graph_vector[i_channel].GetXaxis().SetTitle("time [ns]")
                graph_vector[i_channel].GetYaxis().SetTitle("amplitude [mV]")
                graphTitle = f"S{station_number[0]}, R{run_number[0]}, Evt{event_number[0]}, Ch{channel_id}"
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

    file_out.cd()
    tree_out.Write()
    file_out.Close()

    if isSim:
        print(f"Station {stationNumber}  Energy {sim_E}")
        print("Number of BAD sim events: " + str(nEvents_badSim))
    else:
        print(f"Station {stationNumber}  Run {runNumber}")
        print("Number of forced trigger events: " + str(nEvents_FT))
    print("Number of total RF events: " + str(nEvents_total))
    print("Number of RF events passed the hitFilter: " + str(nEvents_passedHF))
