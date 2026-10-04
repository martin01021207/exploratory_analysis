import os
import argparse
import numpy as np

import ROOT
from ROOT import TFile, TTree

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import MakeVariables
from NuRadioReco.utilities import trace_utilities

nChannels = 24
inIceChannels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_in", type=str)
    parser.add_argument("dir_out", type=str)
    parser.add_argument("station", type=int)
    parser.add_argument("run", type=int)
    parser.add_argument("event", type=int)
    parser.add_argument("targetChannel", type=int)
    args = parser.parse_args()

    dir_in = args.dir_in
    if not dir_in.endswith("/"):
        dir_in += "/"

    dir_out = args.dir_out
    if not dir_out.endswith("/"):
        dir_out += "/"

    station = args.station
    run = args.run
    event = args.event
    targetChannel = args.targetChannel


    file_candidates = [
        dir_in + f"filtered_s{station}_r{run}.root",
        dir_in + f"events_s{station}_r{run}.root"
    ]

    pathToFile = None
    for path in file_candidates:
        if os.path.exists(path):
            pathToFile = path
            break

    if pathToFile is None:
        print(f"Cannot find file for station {station}, run {run}")
        quit()

    inFile = TFile(pathToFile)
    if inFile.IsZombie():
        print(f"Cannot open file: {pathToFile}")
        quit()

    graph_vector = ROOT.std.vector["TGraph"](nChannels)
    inTree = inFile.Get("events")
    inTree.SetBranchAddress("waveform_graphs", ROOT.AddressOf(graph_vector))
    nEvents_total = inTree.GetEntries()

    graphFile = f'eventWFs_s{station}_r{run}_evt{event}.pdf'
    pdf = PdfPages(dir_out+f'{graphFile}')

    for i_event in range(nEvents_total):
        inTree.GetEntry(i_event)

        if inTree.event_number == event:
            print(f"Found Event_{event}!")
            trace = []
            trace_PA = []
            times_PA = []
            trace_inIce = []
            times_inIce = []
            times = []
            RMS = []
            entropy = []

            fig_set, ax_set = plt.subplots(4, 6, figsize=(14, 8))
            for i_channel in range(nChannels):
                wf = graph_vector[i_channel]
                y = np.array( wf.GetY() )
                x = np.array( wf.GetX() )

                trace.append( y )
                times.append( x )

                if i_channel < 4:
                    trace_PA.append(trace[i_channel])
                    times_PA.append(times[i_channel])
                if i_channel in inIceChannels:
                    trace_inIce.append(trace[i_channel])
                    times_inIce.append(times[i_channel])

                RMS.append( trace_utilities.get_split_trace_noise_RMS(y) )
                entropy.append( trace_utilities.get_entropy(y) )

                row = int(i_channel/6)
                col = int(i_channel%6)

                ax_set[row, col].plot(times[i_channel], trace[i_channel], linewidth=0.7, color='k', label='waveform', zorder=0)
                ax_set[row, col].set_xlabel("time [ns]", fontsize = 8.5)
                ax_set[row, col].set_ylabel("amplitude [mV]", fontsize = 8.5)
                ax_set[row, col].tick_params(axis='x', labelsize=7.5)
                ax_set[row, col].tick_params(axis='y', labelsize=7.5)
                ax_set[row, col].minorticks_on()
                graphTitle = f"S{station}, R{inTree.run_number}, Evt{inTree.event_number}, Ch{i_channel}"
                ax_set[row, col].set_title(graphTitle, fontsize=8.5)
                xMin, xMax = ax_set[row, col].get_xlim()
                ax_set[row, col].set_xlim(xMin, xMax)
                ax_set[row, col].hlines(y=trace_utilities.get_split_trace_noise_RMS(trace[i_channel]), xmin=xMin, xmax=xMax, linestyle='--', linewidth=1, color='red', label='noise RMS')
                ax_set[row, col].legend(loc='upper right', fontsize = 'xx-small')

                del wf

            fig_set.tight_layout(w_pad=1.2)
            pdf.savefig(fig_set)
            plt.close(fig_set)
            del fig_set


            fig_set, ax_set = plt.subplots(4, 6, figsize=(14, 8))
            for i_channel in range(nChannels):
                row = int(i_channel/6)
                col = int(i_channel%6)

                y = trace_utilities.get_hilbert_envelope(trace[i_channel])

                ax_set[row, col].plot(times[i_channel], y, linewidth=0.7, color='orange', label='envelope', zorder=0)
                ax_set[row, col].set_xlabel("time [ns]", fontsize = 8.5)
                ax_set[row, col].set_ylabel("amplitude [mV]", fontsize = 8.5)
                ax_set[row, col].tick_params(axis='x', labelsize=7.5)
                ax_set[row, col].tick_params(axis='y', labelsize=7.5)
                ax_set[row, col].minorticks_on()
                graphTitle = f"S{station}, R{inTree.run_number}, Evt{inTree.event_number}, Ch{i_channel}"
                ax_set[row, col].set_title(graphTitle, fontsize=8.5)
                xMin, xMax = ax_set[row, col].get_xlim()
                ax_set[row, col].set_xlim(xMin, xMax)
                ax_set[row, col].hlines(y=trace_utilities.get_split_trace_noise_RMS(y), xmin=xMin, xmax=xMax, linestyle='--', linewidth=1, color='red', label='noise RMS')
                ax_set[row, col].legend(loc='upper right', fontsize = 'xx-small')

            fig_set.tight_layout(w_pad=1.2)
            pdf.savefig(fig_set)
            plt.close(fig_set)
            del fig_set


            fig_set, ax_set = plt.subplots(1, 2, figsize=(12, 6))
            x = times[targetChannel]
            y = trace[targetChannel]
            ax_set[0].plot(x, y, linewidth=0.9, color='k', label='waveform', zorder=0)
            ax_set[1].plot(x, trace_utilities.get_hilbert_envelope(y), linewidth=0.9, color='orange', label='envelope', zorder=0)
            graphTitle = f"S{station}, R{inTree.run_number}, Evt{inTree.event_number}, Ch{targetChannel}"
            for i in range(2):
                ax_set[i].set_xlabel("time [ns]", fontsize = 16.0)
                ax_set[i].set_ylabel("amplitude [mV]", fontsize = 16.0)
                ax_set[i].tick_params(axis='x', labelsize=14.5)
                ax_set[i].tick_params(axis='y', labelsize=14.5)
                ax_set[i].minorticks_on()
                ax_set[i].set_title(graphTitle, fontsize=16.0)
                xMin, xMax = ax_set[i].get_xlim()
                ax_set[i].set_xlim(xMin, xMax)
                ax_set[i].hlines(y=trace_utilities.get_split_trace_noise_RMS(y), xmin=xMin, xmax=xMax, linestyle='--', linewidth=3, color='red', label='noise RMS')
                ax_set[i].legend(loc='upper right', fontsize = 'xx-large')

            fig_set.tight_layout(w_pad=1.2)
            pdf.savefig(fig_set)
            plt.close(fig_set)
            del fig_set


            fig_set, ax_set = plt.subplots(1, 2, figsize=(12, 6))
            entropy = np.array(entropy)
            refIndex_PA, refIndex_inIce, refIndex_surface = MakeVariables.getReferenceTraceIndices(entropy)
            csw_PA = trace_utilities.get_coherent_sum( np.delete(trace_PA, refIndex_PA, axis=0), trace_PA[refIndex_PA] )
            ax_set[0].plot(times_PA[refIndex_PA], csw_PA, linewidth=0.9, label='CSW')
            ax_set[0].plot(times_PA[refIndex_PA], trace_utilities.get_hilbert_envelope(csw_PA), linewidth=0.9, color='orange', label='envelope')
            csw_inIce = trace_utilities.get_coherent_sum( np.delete(trace_inIce, refIndex_inIce, axis=0), trace_inIce[refIndex_inIce] )
            ax_set[1].plot(times_inIce[refIndex_inIce], csw_inIce, linewidth=0.9, label='CSW')
            ax_set[1].plot(times_inIce[refIndex_inIce], trace_utilities.get_hilbert_envelope(csw_inIce), linewidth=0.9, color='orange', label='envelope')

            title = f"S{station}, R{inTree.run_number}, Evt{inTree.event_number}"
            for i in range(2):
                ax_set[i].set_xlabel("time [ns]", fontsize = 16.0)
                ax_set[i].set_ylabel("amplitude [mV]", fontsize = 16.0)
                ax_set[i].tick_params(axis='x', labelsize=14.5)
                ax_set[i].tick_params(axis='y', labelsize=14.5)
                ax_set[i].minorticks_on()
                if i == 0:
                    graphTitle = title + ", PA"
                elif i == 1:
                    graphTitle = title + ", All In-Ice"
                ax_set[i].set_title(graphTitle, fontsize=16.0)
                xMin, xMax = ax_set[i].get_xlim()
                ax_set[i].set_xlim(xMin, xMax)
                ax_set[i].legend(loc='upper right', fontsize = 'large')

            fig_set.tight_layout(w_pad=1.2)
            pdf.savefig(fig_set)
            plt.close(fig_set)
            del fig_set

    inFile.Close()
    pdf.close()
