import os
import numpy as np
import json
import sys
import uproot
from datetime import datetime, timezone
import csv
from collections import defaultdict
import bisect

station_num = int(sys.argv[1])
print(f"Station {station_num}")

# where to save output
dir_out = "/mnt/nrdstor/hep/martinliu/data/realData/triggerRates/"
outFilename = f"highTrigRuns_s{station_num}.txt"

if not os.path.exists(dir_out):
    os.makedirs(dir_out)


def has_high_trigger_rate(trigger_times, window_size=30, rate_threshold=2):
    """Return True if any forward time window reaches the rate threshold.

    ``trigger_times`` must be sorted. ``searchsorted`` performs the same
    [start, start + window_size) counting as the old implementation without
    constructing a full-array mask once per event.
    """
    if trigger_times.size == 0:
        return False

    window_ends = np.searchsorted(
        trigger_times,
        trigger_times + window_size,
        side="left",
    )
    counts = window_ends - np.arange(trigger_times.size)
    return bool(np.any(counts >= rate_threshold * window_size))


def parse_run_times(run_info_path):
    run_start = None
    run_end = None
    with open(run_info_path, 'r') as file:
        for line in file:
            if line.startswith("RUN-START-TIME"):
                run_start = float(line.split('=')[1].strip())
            elif line.startswith("RUN-END-TIME"):
                run_end = float(line.split('=')[1].strip())
    if run_start is None or run_end is None:
        raise ValueError("RUN-START-TIME or RUN-END-TIME not found in file.")
    return run_start, run_end

def read_utc_times(file_path):
    with open(file_path, 'r') as f:
        result = []
        for line in f:
            value = line.strip()
            if not value:
                continue
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            result.append(dt)
        return result


def get_events_in_positive_time_window(event_numbers, trigger_times, timestamps, window_size=60):
    """Return events with timestamp <= trigger_time <= timestamp + window_size."""
    if len(timestamps) == 0 or trigger_times.size == 0:
        return []

    timestamps = np.asarray(timestamps, dtype=float)
    indices = np.searchsorted(timestamps, trigger_times, side="right") - 1
    valid = indices >= 0
    matched = np.zeros(trigger_times.size, dtype=bool)
    matched[valid] = trigger_times[valid] <= timestamps[indices[valid]] + window_size
    return event_numbers[matched].astype(int).tolist()


def get_events_in_symmetric_time_window(event_numbers, trigger_times, timestamps, half_width=150):
    """Return events within half_width seconds before or after a timestamp."""
    if len(timestamps) == 0 or trigger_times.size == 0:
        return []

    timestamps = np.asarray(timestamps, dtype=float)
    indices = np.searchsorted(timestamps, trigger_times - half_width, side="left")
    valid = indices < timestamps.size
    matched = np.zeros(trigger_times.size, dtype=bool)
    matched[valid] = timestamps[indices[valid]] <= trigger_times[valid] + half_width
    return event_numbers[matched].astype(int).tolist()


def merged_interval_duration(intervals, run_start, run_end):
    """Duration of arbitrary intervals inside a run, with overlaps counted once."""
    clipped = sorted(
        (max(start, run_start), min(end, run_end))
        for start, end in intervals
        if max(start, run_start) < min(end, run_end)
    )
    if not clipped:
        return 0.0

    total = 0.0
    current_start, current_end = clipped[0]
    for start, end in clipped[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def count_events_outside_intervals(trigger_times, intervals):
    """Count events not covered by any excluded time interval."""
    if trigger_times.size == 0:
        return 0
    if not intervals:
        return int(trigger_times.size)

    # A single boolean mask gives the union of all exclusions, so an event in
    # overlapping airplane/high-wind windows is still counted only once.
    excluded = np.zeros(trigger_times.size, dtype=bool)
    for start, end in intervals:
        excluded |= (trigger_times >= start) & (trigger_times <= end)
    return int(np.count_nonzero(~excluded))

# Airplane info
timestamp_file = f"/mnt/nrdstor/hep/martinliu/data/realData/triggerRates/airplane_timestamps_s{station_num}.txt"
run_to_timestamps = defaultdict(list)
with open(timestamp_file) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for ts, run in reader:
        run_to_timestamps[int(run)].append(float(ts))

# Sort timestamps for each run
for run in run_to_timestamps:
    run_to_timestamps[run].sort()

matched_events = defaultdict(list)


# read burned sample json to select run/event numbers
dir_burnList = f'/mnt/nrdstor/hep/martinliu/data/realData/burnData/JSON_lists'
burnLists = []
burnLists.append(dir_burnList + f"/station{station_num}_2022_burn_sample_evt_num_dict.json")
#burnLists.append(dir_burnList + f"/station{station_num}_2023_burn_sample_evt_num_dict.json")

utc_times_path = f"/mnt/nrdstor/hep/martinliu/data/realData/triggerRates/high_wind_dates.txt"
utc_times = read_utc_times(utc_times_path)
high_wind_timestamps = sorted(dt.timestamp() for dt in utc_times)
high_wind_events = defaultdict(list)

nRuns = 0
nRuns_highTrig = 0
nRuns_highWind = 0
nEvents_total = 0
nEvents_excludedHighTrigger = 0
nEvents_excludedAirplane = 0
nEvents_excludedHighWind = 0
nEvents_notExcluded = 0
lifeTime_total = 0
lifeTime_excluded = 0
d = np.array([])
for burnList in burnLists:
    with open(burnList, 'r') as jfile:
        data = json.load(jfile)
        first_key = next(iter(data))
        last_key = next(reversed(data))
        run_start = int(first_key)
        run_stop = int(last_key)
        for index, key in enumerate((data.keys())):

            if int(key) >= run_start and int(key) <= run_stop:
                run_num = int(key)

                dir_file = f'/mnt/nrdstor/hep/martinliu/data/rnogData/station{station_num}/run{run_num}'

                # Open the ROOT file using uproot
                with uproot.open(dir_file+"/headers.root") as file:
                    tree = file[str(file.keys()[0])]
                    triggerTimes = np.asarray(
                        tree['header/trigger_time'].array(library="np"),
                        dtype=float,
                    )
                    event_numbers = np.asarray(
                        tree['header/event_number'].array(library="np")
                    )

                nRuns += 1
                nEvents_total += int(triggerTimes.size)

                if triggerTimes.size == 0:
                    print(f'Empty run, Run {run_num}')
                    continue

                # Keep trigger times and event numbers aligned if a file is not
                # already ordered chronologically.
                if np.any(triggerTimes[1:] < triggerTimes[:-1]):
                    order = np.argsort(triggerTimes, kind="stable")
                    triggerTimes = triggerTimes[order]
                    event_numbers = event_numbers[order]

                isHighTriggerRun = has_high_trigger_rate(
                    triggerTimes,
                    window_size=30,
                    rate_threshold=2,
                )

                airplane_times_for_run = run_to_timestamps.get(run_num, [])
                matched_airplane_events = get_events_in_symmetric_time_window(
                    event_numbers,
                    triggerTimes,
                    airplane_times_for_run,
                    half_width=150,
                )
                if matched_airplane_events:
                    matched_events[run_num].extend(matched_airplane_events)

                run_info_path = f"/mnt/nrdstor/hep/martinliu/data/rnogData/station{station_num}/run{run_num}/aux/runinfo.txt"
                try:
                    run_start_time, run_end_time = parse_run_times(run_info_path)
                    lifeTime_run = run_end_time - run_start_time
                except (ValueError, FileNotFoundError):
                    run_start_time = float(triggerTimes[0])
                    run_end_time = float(triggerTimes[-1])
                    lifeTime_run = abs(run_end_time - run_start_time)

                # Select only events from each high-wind timestamp through 60 s later.
                # bisect limits the timestamps checked to those that can affect this run.
                first_wind = bisect.bisect_left(high_wind_timestamps, run_start_time - 60)
                last_wind = bisect.bisect_right(high_wind_timestamps, run_end_time)
                wind_times_for_run = high_wind_timestamps[first_wind:last_wind]

                matched_wind_events = (
                    get_events_in_positive_time_window(
                        event_numbers,
                        triggerTimes,
                        wind_times_for_run,
                        window_size=60,
                    )
                    if wind_times_for_run
                    else []
                )
                if matched_wind_events:
                    high_wind_events[run_num].extend(matched_wind_events)

                lifeTime_total += lifeTime_run

                if isHighTriggerRun:
                    # The later analysis removes this entire run.
                    nEvents_excludedHighTrigger += int(event_numbers.size)
                    nRuns_highTrig += 1
                    lifeTime_excluded += lifeTime_run
                    d = np.append(d, int(run_num))
                    print(f'High Trigger Rate, Run {run_num}')
                else:
                    # Merge all time-based exclusions before adding them so overlaps
                    # between airplane and high-wind windows are not counted twice.
                    exclusion_windows = []

                    exclusion_windows.extend(
                        (timestamp - 150, timestamp + 150)
                        for timestamp in airplane_times_for_run
                    )

                    # Livetime exclusion depends on the wind timestamps, not on
                    # whether an event happened to be recorded during the window.
                    exclusion_windows.extend(
                        (timestamp, timestamp + 60)
                        for timestamp in wind_times_for_run
                    )

                    if wind_times_for_run:
                        nRuns_highWind += 1
                        print(
                            f'High Wind Speed, Run {run_num}: '
                            f'{len(matched_wind_events)} matched events'
                        )

                    # Clip and merge all intervals together.
                    if exclusion_windows:
                        lifeTime_excluded += merged_interval_duration(
                            exclusion_windows,
                            run_start_time,
                            run_end_time,
                        )

                    # Match the later event-removal workflow exactly: remove the
                    # union of event IDs written to the airplane and high-wind
                    # JSON lists. Events matching both criteria are removed once.
                    airplane_event_ids = set(matched_airplane_events)
                    wind_event_ids = set(matched_wind_events)
                    excluded_event_ids = airplane_event_ids | wind_event_ids

                    nEvents_excludedAirplane += int(
                        np.count_nonzero(
                            np.isin(event_numbers, list(airplane_event_ids))
                        )
                    )
                    nEvents_excludedHighWind += int(
                        np.count_nonzero(
                            np.isin(
                                event_numbers,
                                list(wind_event_ids - airplane_event_ids),
                            )
                        )
                    )
                    nEvents_notExcluded += int(
                        np.count_nonzero(
                            ~np.isin(event_numbers, list(excluded_event_ids))
                        )
                    )

np.savetxt(dir_out+outFilename, d, fmt='%d')


# Airplane event info output
output = {str(run): evts for run, evts in matched_events.items()}
with open(f"/mnt/nrdstor/hep/martinliu/data/realData/triggerRates/airplane_events_s{station_num}.json", "w") as f:
    json.dump(output, f, indent=2)


# High-wind event info output. These events occur from each high-wind
# timestamp through 60 seconds afterward; events before the timestamp are not selected.
high_wind_output = {str(run): evts for run, evts in high_wind_events.items()}
with open(f"/mnt/nrdstor/hep/martinliu/data/realData/triggerRates/high_wind_events_s{station_num}.json", "w") as f:
    json.dump(high_wind_output, f, indent=2)


days_lifeTime_total = lifeTime_total / (24 * 60 * 60)
days_lifeTime_excluded = lifeTime_excluded / (24 * 60 * 60)
days_lifeTime_left = (lifeTime_total - lifeTime_excluded) / (24 * 60 * 60)

nEvents_accounted = (
    nEvents_excludedHighTrigger
    + nEvents_excludedAirplane
    + nEvents_excludedHighWind
    + nEvents_notExcluded
)
if nEvents_accounted != nEvents_total:
    raise RuntimeError(
        "Event accounting mismatch: "
        f"{nEvents_accounted} accounted for out of {nEvents_total}"
    )

print(f"Number of Runs: {nRuns}")
print(f"Total Number of Events: {nEvents_total}")
print(f"Events Excluded with High Trigger Rate Runs: {nEvents_excludedHighTrigger}")
print(f"Events Excluded by Airplane Windows: {nEvents_excludedAirplane}")
print(
    "Additional Events Excluded by High Wind Windows: "
    f"{nEvents_excludedHighWind}"
)
print(f"Number of Events after Exclusions: {nEvents_notExcluded}")
print(f"Number of Runs with High Trigger Rate: {nRuns_highTrig}")
print(f"Number of Runs containing selected High Wind events: {nRuns_highWind}")
print(f"Number of selected High Wind events: {sum(len(evts) for evts in high_wind_events.values())}")

print(f"Total Lifetime: {days_lifeTime_total} days")
print(f"Total Lifetime Excluded: {days_lifeTime_excluded} days")
print(f"Total Lifetime after Cut: {days_lifeTime_left} days")
