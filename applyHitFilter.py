import os
import re
import glob
import math
import json
import yaml
import h5py
import argparse
import logging
import datetime
import numpy as np
import pandas as pd
from array import array
from pathlib import Path

from collections import defaultdict

import NuRadioReco.modules.RNO_G.stationHitFilter
import NuRadioReco.modules.RNO_G.channelBlockOffsetFitter
import NuRadioReco.modules.RNO_G.channelGlitchDetector
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelSinewaveSubtraction
import NuRadioReco.modules.interferometricDirectionReconstruction3D
from NuRadioReco.framework.parameters import showerParameters
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units, trace_utilities, logging as nulogging

import ROOT
from ROOT import TFile, TTree, TGraph


logger = logging.getLogger("NuRadioReco.example.RNOG.apply_hit_filter")
logger.setLevel(nulogging.LOGGING_STATUS)


def point_along_line(A, B, dist):
    """
    Return point C that lies on line AB, starting at A,
    going dist meters in the direction from A → B.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    v = B - A
    v_hat = v / np.linalg.norm(v)
    C = A + dist * v_hat
    return C


def point_along_direction(p, theta, phi, d=-10.0, degrees=True):
    """
    p      : starting point (x, y, z)
    theta  : zenith angle (from +z)
    phi    : azimuth angle (from +x toward +y)
    d      : distance to move along direction
    """
    p = np.asarray(p, dtype=float)

    if degrees:
        theta = np.deg2rad(theta)
        phi   = np.deg2rad(phi)

    # Direction unit vector
    v_hat = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    return p + d * v_hat


def get_zenith_azimuth(p1, p2, degrees=True):
    """
    Compute the zenith and azimuth angles of the line from p1 to p2.

    Zenith: angle from +z axis (0° = up, 180° = down)
    Azimuth: angle from +x axis in the x–y plane (counter-clockwise)
    """
    # Convert to numpy arrays
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)

    # Direction vector from p1 to p2
    v = p2 - p1
    v_mag = np.linalg.norm(v)
    if v_mag == 0:
        raise ValueError("The two points are identical; direction is undefined.")

    # --- Zenith (angle from +z axis) ---
    cos_theta = v[2] / v_mag
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # radians

    # --- Azimuth (angle in x–y plane from +x) ---
    phi = np.arctan2(v[1], v[0])  # radians, range (-π, π]

    if degrees:
        theta = np.degrees(theta)
        phi = np.degrees(phi)
        # Convert azimuth to [0, 360)
        if phi < 0:
            phi += 360.0

    return theta, phi


def get_sim_vertex(station_id, event_object, det, is_FAERIE, vertex_direction=None):
    station_pos_abs = np.array(det.get_absolute_position(station_id))
    ch1_pos_rel = np.array(det.get_relative_position(station_id, 1))
    ch2_pos_rel = np.array(det.get_relative_position(station_id, 2))
    pa_pos_rel_station = 0.5 * (ch1_pos_rel + ch2_pos_rel)  # PA center relative to station
    pa_pos_abs = station_pos_abs + pa_pos_rel_station  # PA absolute position
    if is_FAERIE:
        sim_efield = event_object.get_station().get_sim_station().get_electric_fields()[0]
        station_position = np.array(sim_efield.get_position())
        station_position[2] = pa_pos_abs[2]
        showers = [sh for sh in event_object.get_sim_showers()]
        shower_source = np.array(showers[0].get_axis())
        shower_axis = (-1) * shower_source
        shower_core = point_along_line(np.array([0,0,0]), shower_axis, 5)
        interaction_vertex_rel_PA = shower_core - station_position
    else:
        interaction_vertex_abs = list(event_object.get_sim_showers())[0].get_parameter(showerParameters.vertex)
        shower_axis = np.array(interaction_vertex_abs)
        shower_core = point_along_direction(shower_axis, vertex_direction[0], vertex_direction[1])
        interaction_vertex_rel_PA = shower_core - pa_pos_abs

    x_rel_PA, y_rel_PA, z_rel_PA = interaction_vertex_rel_PA
    rho_rel_PA = np.sqrt(x_rel_PA**2 + y_rel_PA**2)
    r_rel_PA = np.sqrt(rho_rel_PA**2 + z_rel_PA**2)
    zenith_rel_PA, azimuth_rel_PA = get_zenith_azimuth(np.array([0,0,0]), interaction_vertex_rel_PA)

    return (r_rel_PA, zenith_rel_PA, azimuth_rel_PA)


def get_interaction_type(interaction_type):
    return {"cc": 0, "nc": 1}[interaction_type]


def get_run_key_from_nur(nur_file):
    base = os.path.basename(nur_file)
    match = re.search(r"(.+_[jc]\d+)", base)
    if not match:
        raise ValueError(f"Could not extract run key from: {nur_file}")
    return match.group(1)


def get_run_number(nur_file):
    base = os.path.basename(nur_file)
    match = re.search(r"_[jc](\d+)", base)
    if not match:
        raise ValueError(f"Could not extract run number from: {nur_file}")
    return int(match.group(1))


event_counter = defaultdict(int)


def get_next_event_id(run):
    event_id = event_counter[run]
    event_counter[run] += 1
    return event_id


def get_csv_from_nur(nur_file):
    run_key = get_run_key_from_nur(nur_file)
    return f"{run_key}_ledger.csv"


def get_hdf5_from_nur(nur_file):
    run_key = get_run_key_from_nur(nur_file)
    return f"{run_key}_ledger.hdf5"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_in", type=str, help="Input directory")
    parser.add_argument("dir_out", type=str, help="Output directory")
    parser.add_argument("station_number", type=int, help="Station number")
    parser.add_argument("run", type=int, help="Run number")
    parser.add_argument('--json_select', type=str, default=None, help="JSON file of events to select")
    parser.add_argument("--isExcluded", action="store_true", help="Exclude events in JSON")
    parser.add_argument("--uproot", action="store_true", help="Using uproot backend")
    parser.add_argument("--isSim", action="store_true", help="Simulation input")
    parser.add_argument("--sim_E", type=str, default=None, help="Simulation energy")
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

    reco_config = "/home/hep/martinliu/research/reconstruction/reco_3D/reco.yaml"
    with open(reco_config, "r") as f:
        config = yaml.safe_load(f)
    config["station_id"] = stationNumber
    config["time_delay_tables"] = "/mnt/nrdstor/hep/martinliu/data/reconstruction/time_delay_tables"
    config["validation"] = True

    isSelectingEvents = False
    if isSim:
        if sim_E is None:
            parser.error("--sim_E is required when --isSim is used.")
        else:
            sim_E = sim_E + "eV" if not sim_E.endswith("eV") else sim_E
            sim_E_number = float( sim_E.replace('eV','') )
        import NuRadioReco.modules.io.eventReader as readSimData
        treename = "events_sim"
        filename_out = f"filtered_sim_s{stationNumber}_{sim_E}_r{runNumber}.root"
        path_dir_in = Path(dir_in)
        run_str = f"{runNumber:04d}"
        fileList = sorted(path_dir_in.glob(f"*_j{run_str}*.nur"))
        if not fileList:
            fileList = sorted(path_dir_in.glob(f"*_c{run_str}*.nur"))
    else:
        if runNumber is None:
            parser.error("Run number is missing, please specify one for the real data!")

        # Skip runs with high trigger rates
        highTrigRuns = np.loadtxt(f"/mnt/nrdstor/hep/martinliu/data/realData/triggerRates/highTrigRuns_s{stationNumber}.txt", dtype=int)
        if runNumber in highTrigRuns:
            print(f"NOT PROCESSED:  Station {stationNumber}  Run {runNumber}")
            quit()

        # Airplane events
        json_airplane_events = f"/mnt/nrdstor/hep/martinliu/data/realData/triggerRates/airplane_events_s{stationNumber}.json"
        with open(json_airplane_events, 'r') as json_file:
            airplane_runs = json.loads(json_file.read())

        # High wind events
        json_high_wind_events = f"/mnt/nrdstor/hep/martinliu/data/realData/triggerRates/high_wind_events_s{stationNumber}.json"
        with open(json_high_wind_events, 'r') as json_file:
            high_wind_runs = json.loads(json_file.read())

        if json_select:
            isSelectingEvents = True
            with open(json_select, 'r') as json_selectEvents:
                selection = json.loads(json_selectEvents.read())
            if str(runNumber) in selection:
                eventList = selection[str(runNumber)]
            else:
                #eventList = []
                #if not isExcluded:
                print(f"Run {runNumber} is not in the list, exit now.")
                quit()
        import NuRadioReco.modules.io.RNO_G.readRNOGDataMattak
        treename = "events"
        filename_out = f"filtered_s{stationNumber}_r{runNumber}.root"
        data = f"station{stationNumber}/run{runNumber}"
        fileList = []
        fileList.append(dir_in + data)

    nChannels = 24
    graph_vector = ROOT.std.vector["TGraph"](nChannels)

    station_number = array('i', [0])
    run_number = array('i', [0])
    event_number = array('i', [0])

    sim_energy = array('f', [0.])
    shower_energy = array('f', [0.])
    inelasticity = array('f', [0.])
    interaction_type = array('i', [0])

    trigger_time = array('d', [0.])

    true_radius = array('f', [0.])
    true_theta = array('f', [0.])
    true_phi = array('f', [0.])
    true_source_theta = array('i', [0])
    true_source_phi = array('i', [0])

    reco_max_corr = array('f', [np.nan])
    reco_surf_corr_z = array("f", [np.nan])
    reco_surf_corr_zen = array("f", [np.nan])
    reco_rho = array('f', [np.nan])
    reco_phi = array('f', [np.nan])
    reco_z = array('f', [np.nan])

    passed_hit_filter = array('i', [0])
    nCoincidentPairs_PA = array('i', [0])
    nHighHits_PA = array('i', [0])
    nCoincidentPairs_inIce = array('i', [0])
    nHighHits_inIce = array('i', [0])


    tree_out = ROOT.TTree(treename, treename)

    file_out = TFile(dir_out+filename_out, "RECREATE")

    tree_out.Branch("waveform_graphs", "std::vector<TGraph>", graph_vector)

    tree_out.Branch("station_number", station_number, 'station_number/I')
    tree_out.Branch("run_number", run_number, 'run_number/I')
    tree_out.Branch("event_number", event_number, 'event_number/I')

    tree_out.Branch("sim_energy", sim_energy, 'sim_energy/F')
    tree_out.Branch("shower_energy", shower_energy, 'shower_energy/F')
    tree_out.Branch("inelasticity", inelasticity, 'inelasticity/F')
    tree_out.Branch("interaction_type", interaction_type, 'interaction_type/I')

    tree_out.Branch("trigger_time", trigger_time, 'trigger_time/D')

    tree_out.Branch("true_radius", true_radius, 'true_radius/F')
    tree_out.Branch("true_theta", true_theta, 'true_theta/F')
    tree_out.Branch("true_phi", true_phi, 'true_phi/F')
    tree_out.Branch("true_source_theta", true_source_theta, 'true_source_theta/I')
    tree_out.Branch("true_source_phi", true_source_phi, 'true_source_phi/I')

    tree_out.Branch("reco_max_corr", reco_max_corr, 'reco_max_corr/F')
    tree_out.Branch("reco_surf_corr_z", reco_surf_corr_z, 'reco_surf_corr_z/F')
    tree_out.Branch("reco_surf_corr_zen", reco_surf_corr_zen, 'reco_surf_corr_zen/F')
    tree_out.Branch("reco_rho", reco_rho, 'reco_rho/F')
    tree_out.Branch("reco_phi", reco_phi, 'reco_phi/F')
    tree_out.Branch("reco_z", reco_z, 'reco_z/F')

    tree_out.Branch("passed_hit_filter", passed_hit_filter, 'passed_hit_filter/I')
    tree_out.Branch("nCoincidentPairs_PA", nCoincidentPairs_PA, 'nCoincidentPairs_PA/I')
    tree_out.Branch("nHighHits_PA", nHighHits_PA, 'nHighHits_PA/I')
    tree_out.Branch("nCoincidentPairs_inIce", nCoincidentPairs_inIce, 'nCoincidentPairs_inIce/I')
    tree_out.Branch("nHighHits_inIce", nHighHits_inIce, 'nHighHits_inIce/I')

    tree_out.SetDirectory(file_out)
    tree_out.SetMaxTreeSize(1000000000000)

    det = detector.Detector(source="rnog_mongo")
    det.update(datetime.datetime(2022, 10, 1))

    stationHitFilter = NuRadioReco.modules.RNO_G.stationHitFilter.stationHitFilter(complete_time_check=True, complete_hit_check=True)
    stationHitFilter.begin()

    channelBlockOffsets = NuRadioReco.modules.RNO_G.channelBlockOffsetFitter.channelBlockOffsets()
    channelBlockOffsets.begin()

    #channelGlitchDetector = NuRadioReco.modules.RNO_G.channelGlitchDetector.channelGlitchDetector(cut_value=0.0)
    #channelGlitchDetector.begin()

    channelCableDelayAdder = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
    channelCableDelayAdder.begin()

    #hardwareResponse = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
    #hardwareResponse.begin()

    channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
    channelResampler.begin()

    channelSinewaveSubtraction = NuRadioReco.modules.channelSinewaveSubtraction.channelSinewaveSubtraction()
    channelSinewaveSubtraction.begin(save_filtered_freqs=False, freq_band=(0.1, 0.6))

    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
    channelBandPassFilter.begin()

    reco = NuRadioReco.modules.interferometricDirectionReconstruction3D.InterferometricReco3D()
    reco.begin(station_id=stationNumber, config=config, det=det)

    runNumber_sim = 0
    nEvents_total = 0
    nEvents_FT = 0
    nEvents_RADIANT = 0
    nEvents_PPS = 0
    nEvents_UNKNOWN = 0
    nEvents_bad = 0
    nEvents_badSim = 0
    nEvents_LT = 0
    nEvents_passedHF = 0
    for file in fileList:
        isFAERIE = False
        isNew = False
        if isSim:
            filename = str(file).split("/")[-1]
            if "cos" in filename and "phi" in filename:
                pattern = r"cos_(-?\d+(?:\.\d+)?)-phi_(-?\d+(?:\.\d+)?)"
                m = re.search(pattern, filename)
                cosine, phi = map(float, m.groups())
                source_theta = np.rad2deg(np.arccos(cosine))
                source_phi = phi
                direction = (source_theta, source_phi)
            elif "SIM" in filename:
                isFAERIE = True
                direction = None
                runNumber_sim = int(filename.split(".nur")[0].split("_")[-1])
            else:
                isNew = True
                file_csv = get_csv_from_nur(filename)
                #file_hdf5 = get_hdf5_from_nur(filename)
                ledger = pd.read_csv(dir_in + file_csv)
                runNumber_sim = get_run_number(filename)
            eventID_sim = 0
            reader = readSimData.eventReader()
            reader.begin(file)
        else:
            reader = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData()
            reader.begin(file, apply_baseline_correction=None, mattak_kwargs={'backend': backend})
            info = reader.get_events_information(keys=["station", "run", "eventNumber", "triggerType", "triggerTime"])

        eventCollection = list(reader.run())
        nTotalEventsPerFile = len(eventCollection)
        nEvents_total += nTotalEventsPerFile
        for i_event, event in enumerate(reader.run()):
            station = event.get_station()
            station_id = station.get_id()
            station_number[0] = station_id

            isBadSimEvent = False

            if isSim:
                if not station.has_triggered():
                    continue
                run_number[0] = runNumber_sim
                event_number[0] = eventID_sim
                sim_energy[0] = sim_E_number
                trigger_time[0] = -1.0
                if isNew:
                    event_group_id = event.get_run_number()
                    row = ledger.loc[ledger["event_group_id"] == event_group_id].iloc[0]
                    source_theta = row["zenith_deg"]
                    source_phi = row["azimuth_deg"]
                    direction = (source_theta, source_phi)
                    sim_energy[0] = row["energy_eV"]
                    shower_energy[0] = row["shower_energy_eV"]
                    inelasticity[0] = row["inelasticity"]
                    interaction_type[0] = get_interaction_type(row["interaction_type"])
                    eventID_sim = get_next_event_id(runNumber_sim)
                else:
                    shower_energy[0] = -1.0
                    inelasticity[0] = -1.0
                    interaction_type[0] = -1
                    eventID_sim += 1

                radius, theta, phi = get_sim_vertex(station_id, event, det, isFAERIE, direction)
                true_radius[0] = radius
                true_theta[0] = theta
                true_phi[0] = phi
                if isFAERIE:
                    showers = [sh for sh in  event.get_sim_showers()]
                    shower_source = np.array(showers[0].get_axis())
                    source_theta, source_phi = get_zenith_azimuth(np.array([0,0,0]), shower_source)
                true_source_theta[0] = int(source_theta)
                true_source_phi[0] = int(source_phi)
            else:
                trigger_type = info[i_event].get('triggerType')
                if trigger_type == "FORCE":
                    nEvents_FT += 1
                    continue
                elif "RADIANT" in trigger_type:
                    nEvents_RADIANT += 1
                    continue
                elif "PPS" in trigger_type:
                    nEvents_PPS += 1
                    continue
                elif trigger_type != "LT":
                    nEvents_UNKNOWN += 1
                    continue

                run_number[0] = info[i_event].get('run')
                event_number[0] = info[i_event].get('eventNumber')
                sim_energy[0] = -1.0
                shower_energy[0] = -1.0
                inelasticity[0] = -1.0
                interaction_type[0] = -1

                if isSelectingEvents:
                    if isExcluded:
                        if event_number[0] in eventList:
                            continue
                    else:
                        if event_number[0] not in eventList:
                            continue

                if str(run_number[0]) in airplane_runs:
                    airplane_events = airplane_runs[str(run_number[0])]
                    if int(event_number[0]) in airplane_events:
                        continue

                if str(run_number[0]) in high_wind_runs:
                    high_wind_events = high_wind_runs[str(run_number[0])]
                    if int(event_number[0]) in high_wind_events:
                        continue

                ti = info[i_event].get('triggerTime')
                try:
                    ti = float(ti)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(ti):
                    continue
                trigger_time[0] = ti
                true_radius[0] = -1.0
                true_theta[0] = -1.0
                true_phi[0] = -1.0
                true_source_theta[0] = -1
                true_source_phi[0] = -1

            channelBlockOffsets.run(event, station, det)

            channelCableDelayAdder.run(event, station, det, mode='subtract')

            channelResampler.run(event, station, det, sampling_rate=10 * units.GHz)

            channelSinewaveSubtraction.run(event, station, det, algorithm="sliding", peak_prominence=4.0)

            channelBandPassFilter.run(
                event, station, det,
                passband=(0.1 * units.GHz, 0.7 * units.GHz),
                filter_type='butter', order=10)

            traces = []
            times = []
            channels = np.array([])
            for i_channel, channel in enumerate(station.iter_channels()):
                channel_id = channel.get_id()
                y = np.array(channel.get_trace()) / units.mV
                x = np.array(channel.get_times())

                if isSim:
                    isBadTrace, _, _ = trace_utilities.is_NAN_or_INF(y)
                    if isBadTrace:
                        isBadSimEvent = True

                traces.append(y)
                times.append(x)
                channels = np.append(channels, channel_id)

            if isBadSimEvent:
                nEvents_badSim += 1

            if len(channels) != nChannels:
                print(f"S{station_number[0]} R{run_number[0]} Evt{event_number[0]} does not have 24 channels...")
                nEvents_bad += 1
                continue
            else:
                nEvents_LT += 1

                # Hit Filter
                is_passed_HF = stationHitFilter.run(event, station, det)
                in_time_window = stationHitFilter.is_in_time_window()
                over_hit_threshold = stationHitFilter.is_over_hit_threshold()

                passed_hit_filter[0] = int(is_passed_HF)
                nEvents_passedHF += int(is_passed_HF)

                nCoincidentPairs_PA[0] = int(sum(in_time_window[0]))
                nHighHits_PA[0] = int(sum(over_hit_threshold[:4]))

                nCoincidentPairs_inIce[0] = nCoincidentPairs_PA[0] + int(
                    sum(in_time_window[grp][0] for grp in range(1, 4))
                )
                nHighHits_inIce[0] = int(sum(over_hit_threshold[:15]))


                traces = np.array(traces)
                times = np.array(times)
                sorted_indices = channels.argsort()
                sorted_channels = np.sort(channels)
                traces = traces[sorted_indices]
                times = times[sorted_indices]

                for i_channel in sorted_channels:
                    i_channel = int(i_channel)
                    if isSim:
                        if isFAERIE or isNew:
                            graphTitle = f"{sim_energy[0]}, R{run_number[0]}, Evt{event_number[0]}, Ch{i_channel}"
                        else:
                            # Roll waveform to place the pulse near the center
                            #traces[i_channel] = np.roll(traces[i_channel], 800)
                            graphTitle = f"({sim_energy[0]},{cosine},{int(phi)}): Evt{event_number[0]}, Ch{i_channel}"
                    else:
                        graphTitle = f"S{station_number[0]}, R{run_number[0]}, Evt{event_number[0]}, Ch{i_channel}"
                    graph_vector[i_channel] = TGraph(len(times[i_channel]), times[i_channel], traces[i_channel])
                    graph_vector[i_channel].GetXaxis().SetTitle("time [ns]")
                    graph_vector[i_channel].GetYaxis().SetTitle("amplitude [mV]")
                    graph_vector[i_channel].SetTitle(graphTitle)

                reco_result = reco.run(event, station, det, config)
                reco_rho[0] = float(reco_result.get("rho", np.nan))
                reco_phi[0] = float(reco_result.get("phi", np.nan))
                reco_z[0] = float(reco_result.get("z", np.nan))
                reco_max_corr[0] = float(reco_result.get("max_corr", np.nan))
                reco_surf_corr_z[0] = float(reco_result.get("surf_corr_z", np.nan))
                reco_surf_corr_zen[0] = float(reco_result.get("surf_corr_zen", np.nan))

                tree_out.Fill()

        reader.end()
        if isSim and not isFAERIE and not isNew:
            runNumber_sim += 1

    reco.end()

    file_out.cd()
    tree_out.Write()
    file_out.Close()

    if isSim:
        print(f"Station {stationNumber}  Energy {sim_E}")
        if nEvents_badSim:
            print(f"Number of BAD sim events: {nEvents_badSim}")
    else:
        print(f"Station {stationNumber}  Run {runNumber}")
        print(f"Number of total events: {nEvents_total}")
        print(f"Number of forced trigger events: {nEvents_FT}")
        if nEvents_RADIANT:
            print(f"Number of RADIANT trigger events: {nEvents_RADIANT}")
        if nEvents_PPS:
            print(f"Number of PPS trigger events: {nEvents_PPS}")
        if nEvents_UNKNOWN:
            print(f"Number of UNKNOWN trigger events: {nEvents_UNKNOWN}")
        if nEvents_bad:
            print(f"Number of bad events: {nEvents_bad}")
    if isSelectingEvents:
        print(f"Number of selected LT events: {nEvents_LT}")
    else:
        print(f"Number of total LT events: {nEvents_LT}")
    print(f"Number of LT events passed the hitFilter: {nEvents_passedHF}")
