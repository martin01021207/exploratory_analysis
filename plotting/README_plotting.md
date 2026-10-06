# RNO-G plotting scripts

Run from the project directory using your Python environment with NumPy, Matplotlib, uproot, PyROOT, and NuRadioReco. Replace the example paths and event numbers as needed.

```bash
cd /path/to/project_LDA
mkdir -p analysis/plots
```

## 1. Plot candidate events within their runs

`plot_candidate_runs.py` plots eight analysis variables against trigger time, highlighting candidate events. Use a variables ROOT file containing the runs of interest and a JSON file mapping run numbers to event numbers, for example `{"105": [42, 57]}`.

```bash
python plot_candidate_runs.py \
  analysis/merged/vars_s13_bkg.root /path/to/candidates.json \
  --output analysis/plots/candidate_runs.pdf
```

The PDF has one page per candidate. Add `--combine-events-per-run` for one page per run, or `--relative-time` to show time since the run's first trigger. Use `--tree vars_bkg` if the input contains multiple matching trees.

## 2. Plot a single event's waveforms

`plotSingleEvent.py` plots all channel waveforms and envelopes, a selected channel, and coherent sums.

On Linux, change its `import MakeVariables` line to `import makeVariables as MakeVariables` to match the supplied filename.

```bash
python plotSingleEvent.py analysis/filtered_data analysis/plots 13 105 42 0
```

Arguments: **input directory, output directory, station, run, event, channel** (0–23).

The example reads `filtered_s13_r105.root` (or `events_s13_r105.root`) and writes `analysis/plots/eventWFs_s13_r105_evt42.pdf`. Use the waveform ROOT files from `applyHitFilter.py`.

## 3. Plot BDT variable distributions

`plot_BDT_variable_distributions_train_test_dirs.py` combines training and testing events within each population, then compares background and simulation in a two-page PDF of 13 variables.

```bash
python burn/plot_BDT_variable_distributions_train_test_dirs.py \
  --train-dir analysis/samples/train \
  --test-dir analysis/samples/test \
  --station 13 \
  --output analysis/plots/BDT_variable_distributions_s13.pdf
```

The directories must contain `vars_s13_bkg_train.root` and `vars_s13_sig_train.root`, and the corresponding `_test.root` files, respectively.

Plots use unweighted event counts and a logarithmic y-axis. Add `--linear-y` for a linear axis; use `--bins` and `--bkg-bins` to change histogram bin counts.
