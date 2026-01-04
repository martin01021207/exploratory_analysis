import argparse
import logging
import datetime
import numpy as np
from array import array

import NuRadioReco.modules.RNO_G.stationHitFilter
import NuRadioReco.modules.channelAddCableDelay
from NuRadioReco.framework.parameters import showerParameters
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_in", type=str, help="Input directory")
    parser.add_argument("dir_out", type=str, help="Output directory")
    parser.add_argument("station_number", type=int, help="Station number")
    parser.add_argument("--run", type=int, default=None, help="Run number")
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

    isSelectingEvents = False
    if isSim:
        if sim_E is None:
            parser.error("--sim_E is required when --isSim is used.")
        else:
            sim_E = sim_E + "eV" if not sim_E.endswith("eV") else sim_E
            sim_E_number = float( sim_E.replace('eV','') )
        import re
        import glob
        import NuRadioReco.modules.io.eventReader as readSimData
        treename = "events_sim"
        filename_out = f"filtered_sim_s{stationNumber}_{sim_E}.root"
        fileList = glob.glob(dir_in + "*.nur")
    else:
        if runNumber is None:
            parser.error("Run number is missing, please specify one for the real data!")

        # Skip runs with high trigger rates
        #highTrigRuns = np.loadtxt(f"/mnt/nrdstor/hep/martinliu/data/realData/triggerRates/highTrigRuns_s{stationNumber}.txt", dtype=int)
        #if runNumber in highTrigRuns:
            #print(f"NOT PROCESSED:  Station {stationNumber}  Run {runNumber}")
            #quit()

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
        import NuRadioReco.modules.io.RNO_G.readRNOGDataMattak
        import NuRadioReco.modules.channelResampler
        import NuRadioReco.modules.channelBandPassFilter
        import NuRadioReco.modules.channelSinewaveSubtraction
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
    trigger_time_difference = array('f', [0.])
    true_radius = array('f', [0.])
    true_theta = array('f', [0.])
    true_phi = array('f', [0.])
    true_source_theta = array('i', [0])
    true_source_phi = array('i', [0])

    tree_out = ROOT.TTree(treename, treename)

    file_out = TFile(dir_out+filename_out, "RECREATE")

    tree_out.Branch("waveform_graphs", "std::vector<TGraph>", graph_vector)
    tree_out.Branch("station_number", station_number, 'station_number/I')
    tree_out.Branch("run_number", run_number, 'run_number/I')
    tree_out.Branch("event_number", event_number, 'event_number/I')
    tree_out.Branch("sim_energy", sim_energy, 'sim_energy/F')
    tree_out.Branch("trigger_time_difference", trigger_time_difference, 'trigger_time_difference/F')
    tree_out.Branch("true_radius", true_radius, 'true_radius/F')
    tree_out.Branch("true_theta", true_theta, 'true_theta/F')
    tree_out.Branch("true_phi", true_phi, 'true_phi/F')
    tree_out.Branch("true_source_theta", true_source_theta, 'true_source_theta/I')
    tree_out.Branch("true_source_phi", true_source_phi, 'true_source_phi/I')
    tree_out.SetDirectory(file_out)
    tree_out.SetMaxTreeSize(1000000000000)

    det = detector.Detector(source="rnog_mongo")
    det.update(datetime.datetime(2022, 10, 1))

    channelCableDelayAdder = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
    channelCableDelayAdder.begin()

    # Initialize Hit Filter
    stationHitFilter = NuRadioReco.modules.RNO_G.stationHitFilter.stationHitFilter()
    stationHitFilter.begin()

    runNumber_sim = 0
    nEvents_total = 0
    nEvents_FT = 0
    nEvents_RADIANT = 0
    nEvents_PPS = 0
    nEvents_UNKNOWN = 0
    nEvents_badSim = 0
    nEvents_LT = 0
    nEvents_passedHF = 0
    for file in fileList:
        isFAERIE = False
        if isSim:
            filename = file.split("/")[-1]
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
            eventID_sim = 0
            reader = readSimData.eventReader()
            reader.begin(file)
        else:
            reader = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData()
            reader.begin(file, mattak_kwargs={'backend': backend})
            channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
            channelResampler.begin()
            channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
            channelBandPassFilter.begin()
            channelSinewaveSubtraction = NuRadioReco.modules.channelSinewaveSubtraction.channelSinewaveSubtraction()
            channelSinewaveSubtraction.begin(save_filtered_freqs=False, freq_band=(0.1, 0.7))
            info = reader.get_events_information(keys=["station", "run", "eventNumber", "triggerType", "triggerTime"])

        eventCollection = list(reader.run())
        nTotalEventsPerRun = len(eventCollection)
        nEvents_total += nTotalEventsPerRun

        for i_event, event in enumerate(reader.run()):
            station = event.get_station()
            station_id = station.get_id()

            channelCableDelayAdder.run(event, station, det, mode='subtract')

            isBadSimEvent = False

            station_number[0] = station_id
            if isSim:
                if not station.has_triggered():
                    continue
                run_number[0] = runNumber_sim
                event_number[0] = eventID_sim
                sim_energy[0] = sim_E_number
                trigger_time_difference[0] = 0.
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
                eventID_sim += 1
            else:
                run_number[0] = info[i_event].get('run')
                event_number[0] = info[i_event].get('eventNumber')
                sim_energy[0] = 0.
                time_difference = info[i_event].get('triggerTime') - info[0].get('triggerTime')
                trigger_time_difference[0] = float(time_difference)
                true_radius[0] = 0.
                true_theta[0] = 0.
                true_phi[0] = 0.
                true_source_theta[0] = 0
                true_source_phi[0] = 0
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

                if isSelectingEvents:
                    if isExcluded:
                        if event_number[0] in eventList:
                            continue
                    else:
                        if event_number[0] not in eventList:
                            continue
                channelResampler.run(event, station, det, sampling_rate=5 * units.GHz)
                channelBandPassFilter.run(
                    event, station, det,
                    passband=[0.1 * units.GHz, 0.6 * units.GHz],
                    filter_type='butter', order=10)
                channelSinewaveSubtraction.run(event, station, det, peak_prominence=4.0)

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
                        break

                traces.append(y)
                times.append(x)
                channels = np.append(channels, channel_id)

            if isBadSimEvent:
                nEvents_badSim += 1
                continue

            nEvents_LT += 1

            if len(channels) != nChannels:
                print(f"S{station_number[0]} R{run_number[0]} Evt{event_number[0]} does not have 24 channels...")
                continue
            else:
                # Hit Filter
                is_passed_HF = stationHitFilter.run(event, station, det)
                if not is_passed_HF:
                    continue
                else:
                    nEvents_passedHF += 1

                    traces = np.array(traces)
                    times = np.array(times)
                    sorted_indices = channels.argsort()
                    sorted_channels = np.sort(channels)
                    traces = traces[sorted_indices]
                    times = times[sorted_indices]

                    for i_channel in sorted_channels:
                        i_channel = int(i_channel)
                        if isSim:
                            if isFAERIE:
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

                    tree_out.Fill()

        reader.end()
        if isSim and not isFAERIE:
            runNumber_sim += 1

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
    if isSelectingEvents:
        print(f"Number of selected LT events: {nEvents_LT}")
    else:
        print(f"Number of total LT events: {nEvents_LT}")
    print(f"Number of LT events passed the hitFilter: {nEvents_passedHF}")
