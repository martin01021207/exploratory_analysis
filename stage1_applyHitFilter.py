import argparse
import numpy as np
from array import array

import ROOT
from ROOT import TFile, TTree, TGraph
from HitFilter import HitFilter
import WaveformAnalysisTools as WfAT

nChannels = 24


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_in", type=str, help="Input directory")
    parser.add_argument("dir_out", type=str, help="Output directory")
    parser.add_argument("station", type=int, help="Station number")
    parser.add_argument("--run", type=int, help="Run number")
    parser.add_argument('--json_select', type=str, default=None, help="JSON file of events to select")
    parser.add_argument("--isExcluded", action="store_true", help="Exclude events in JSON")
    parser.add_argument("--uproot", action="store_true", help="Using uproot backend")
    parser.add_argument("--isSim", action="store_true", help="Simulation input")
    parser.add_argument("--sim_E", type=str, help="Simulation energy")
    args = parser.parse_args()

    dir_in = args.dir_in
    if not dir_in.endswith("/"):
        dir_in += "/"

    dir_out = args.dir_out
    if not dir_out.endswith("/"):
        dir_out += "/"

    station = args.station
    run = args.run

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
        if str(run) in selection:
            eventList = selection[str(run)]
        else:
            eventList = []
            if not isExcluded:
                print(f"Run {run} is not in the list, exit now.")
                quit()

    fileList = []

    if isSim:
        import os
        import NuRadioReco.modules.io.eventReader as readSimData
        treename = "events_sim"
        filename_out = f"filtered_sim_s{station}_{sim_E}.root"
        for data in os.listdir(dir_in):
            if data.endswith(".nur"):
                fileList.append(dir_in+str(data))
    else:
        import NuRadioReco.modules.io.RNO_G.readRNOGDataMattak as readRNOGDataMattak
        treename = "events"
        filename_out = f"filtered_s{station}_r{run}.root"
        data = f"station{station}/run{run}/"
        fileList.append(dir_in+data)

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

    nEvents_badSim = 0
    nEvents_FT = 0
    nEvents_total = 0
    nEvents_passedHF = 0
    for file in fileList:
        if isSim:
            run = int(file.split("/")[-1].split(".nur")[0].split("_")[2])
            eventID_sim = 0
            reader = readSimData.eventReader()
            reader.begin(file)
        else:
            reader = readRNOGDataMattak.readRNOGData()
            reader.begin(file, mattak_kwargs={'backend':backend})
            info = reader.get_events_information(keys=["station", "run", "eventNumber", "triggerType"])

        for i_event, event in enumerate(reader.run()):
            station = event.get_station()
            station_id = station.get_id()

            isBadSimEvent = False
            isFT = False

            station_number[0] = station_id
            if isSim:
                run_number[0] = run
                event_number[0] = eventID_sim
                sim_energy[0] = sim_E_number
                eventID_sim += 1
            else:
                run_number[0] = info[i_event].get('run')
                event_number[0] = info[i_event].get('eventNumber')
                sim_energy[0] = 0.
                if info[i_event].get('triggerType') == "FORCE":
                    isFT = True

            if isSelectingEvents:
                if isExcluded:
                    if event_number[0] in eventList:
                        continue
                else:
                    if event_number[0] not in eventList:
                        continue

            HF = HitFilter()
            trace = []
            times = []
            RMS = []

            for i_channel, channel in enumerate(station.iter_channels()):
                channel_id = channel.get_id()
                y = np.array(channel.get_trace())
                x = np.array(channel.get_times())

                if isSim:
                    if WfAT.isBadWaveform(y):
                        isBadSimEvent = True
                        break

                trace.append(y)
                times.append(x)
                RMS.append( WfAT.getNoiseRMS(y) )

                graph_vector[i_channel] = TGraph(len(x), x, y)
                graph_vector[i_channel].GetXaxis().SetTitle("time [ns]")
                graph_vector[i_channel].GetYaxis().SetTitle("amplitude [mV]")
                graphTitle = f"S{station_number[0]}, R{run_number[0]}, Evt{event_number[0]}, Ch{channel_id}"
                graph_vector[i_channel].SetTitle(graphTitle)

            if isBadSimEvent:
                nEvents_badSim += 1
                continue
            elif isFT:
                nEvents_FT += 1
                continue

            passed_HF = HF.passedHitFilter(np.array(trace), np.array(times), np.array(RMS))

            nEvents_total += 1
            if passed_HF:
                nEvents_passedHF += 1
                tree_out.Fill()

            del HF

        if isSim:
            reader.end()

    file_out.cd()
    tree_out.Write()
    file_out.Close()

    print(f"Station {station}")
    if isSim:
        print("Number of BAD sim events: " + str(nEvents_badSim))
    else:
        print("Number of forced trigger events: " + str(nEvents_FT))
    print("Number of total RF events: " + str(nEvents_total))
    print("Number of RF events passed the hitFilter: " + str(nEvents_passedHF))
