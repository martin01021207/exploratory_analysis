import numpy as np
import matplotlib.pyplot as plt
import sys

from astropy.time import Time
from NuRadioReco.utilities import units
from empcart.utilities.plot_helpers import draw_circles, draw_coloraxis_time, draw_stations
from empcart.flight_data.flight_trajectories import FlightTrajectories
# get preprocessed multi-station coincidences
from empcart.config import get_event_cluster_data
clusters = get_event_cluster_data()

from rnog_data.runtable import RunTable

station_number = 13
run_start = 95
run_end = 1084


#station_number = 23
#run_start = 1
#run_end = 1135


rt = RunTable()
table = rt.get_table(stations=[station_number], runs=list(range(run_start, run_end+1)))

ft = FlightTrajectories()

with open(f"/mnt/nrdstor/hep/martinliu/data/realData/triggerRates/airplane_timestamps_s{station_number}.txt", "w") as f:
    f.write("timestamp,run\n")  # Header, darling

    for _, row in table.iterrows():
        start_time = Time(row.time_start)
        end_time = Time(row.time_end)

        ft.update_time(start_time, end_time)
        times = np.linspace(start_time.unix, end_time.unix, 10000)

        for traj in ft.trajectories:
            if not traj.sends_position_data():
                continue

            x, y, z = traj.get_xyz(times)
            mask = (abs(x) < 50 * units.km) & (abs(y) < 50 * units.km)

            if np.sum(mask) == 0:
                continue

            timestamp = traj.get_closest_approach_time(row.station)
            f.write(f"{timestamp},{row.run}\n")
            print(f"Logged airplane at {timestamp} for run {row.run}")
