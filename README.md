# RNO-G analysis workflow

Run from the project directory using your RNO-G Python environment with ROOT/TMVA (`hadd`) and the `simulation_weighting` module required by `testBDT.py`. Examples use station **13**; replace paths and station/run numbers as needed.

## Before running

- In `applyHitFilter.py`, update the reconstruction YAML, time-delay tables, and data-selection file paths (lines 217, 221, 248, 254, 259).
- In the Bash/Slurm scripts, update input/output paths, project/environment paths, log directory, email, and cluster settings. Set `STATION`, `CHUNK`, and `YEARS` or `ENERGIES`. Point `SLURM_SCRIPT` to the matching file in `slurm_scripts/`.
- In `splitEventData.py`, change line 10 to `import makeVariables as MakeVariables` to match the filename on Linux.
- In `estimate_background.py`, set `whole_test_factor` and `burn_test_factor` to your sample fractions (currently `0.93` and `0.03`).

The examples below use these output directories; configure the batch scripts with **absolute paths** to the same locations:

```bash
cd /path/to/project_LDA
mkdir -p analysis/{filtered_data,filtered_sim,vars_data,vars_sim,merged,samples,results,background}
```

## 1. Filter and reconstruct

`applyHitFilter.py` preprocesses waveforms, applies the hit filter and directional reconstruction, and saves waveforms, event identifiers, and reconstruction results to a ROOT file.

```bash
# Recorded data: /path/to/raw contains station13/run105/.
python applyHitFilter.py /path/to/raw analysis/filtered_data 13 105 \
  --json_select /path/to/burn_events.json

# Simulation: directory contains matching .nur files and their CSV ledger.
python applyHitFilter.py /path/to/sim/lgE18.0 analysis/filtered_sim 13 1 \
  --isSim --sim_E 18.0
```

The recorded-data example selects burn-sample events. Omit `--json_select` to process without that selection, or add `--isExcluded` to exclude the listed events.

## 2. Calculate variables

`makeVariables.py` reads a filtered ROOT file and writes analysis variables to a new ROOT file.

```bash
python makeVariables.py analysis/filtered_data/filtered_s13_r105.root analysis/vars_data
python makeVariables.py analysis/filtered_sim/filtered_sim_s13_18.0eV_r1.root analysis/vars_sim
```

### Multiple runs in parallel

Use the `.sh` launchers to submit Slurm arrays. Each task processes `CHUNK` runs; tasks run in parallel. For burn-sample background, set `FULL="false"` in both recorded-data launchers (`true` excludes burn-sample events).

```bash
bash slurm_scripts/applyHitFilter_chunks.sh       # Recorded data
bash slurm_scripts/applyHitFilter_CR_proxy.sh     # Simulation
```

**Wait for filtering to finish**, then submit variable generation:

```bash
bash slurm_scripts/makeVariables_chunks.sh
bash slurm_scripts/makeVariables_CR_proxy.sh
```

Use `squeue -u "$USER"` to monitor jobs. Wait for variable generation to finish before merging files.

## 3. Combine runs with hadd

Merge background and signal separately. Keep the filenames below because the training script expects them. `-f` overwrites an existing merged file.

```bash
hadd -f analysis/merged/vars_s13_bkg.root analysis/vars_data/vars_s13_r*.root
hadd -f analysis/merged/vars_s13_sig.root analysis/vars_sim/vars_sim_s13_*_r*.root
```

## 4. Split into training and testing samples

`--division 2` puts every other event into training and the remainder into testing (approximately 50/50).

```bash
python splitEventData.py analysis/merged/vars_s13_bkg.root analysis/samples --division 2
python splitEventData.py analysis/merged/vars_s13_sig.root analysis/samples --division 2
```

This creates `vars_s13_{bkg,sig}_train.root` under `analysis/samples/train/` and `vars_s13_{bkg,sig}_test.root` under `analysis/samples/test/`.

## 5. Train the BDT

`trainBDT.py` reads both training and testing samples from the supplied directory.

```bash
python burn/trainBDT.py analysis/samples 13
```

The trained weights are saved in `dataLoader_vars_s13/weights/`.

## 6. Test and estimate background

Apply the trained BDT to the background and signal testing samples:

```bash
python burn/testBDT.py 13 \
  analysis/samples/test/vars_s13_bkg_test.root \
  analysis/samples/test/vars_s13_sig_test.root \
  dataLoader_vars_s13/weights analysis/results --target_cut 0.45
```

Then pass its scored testing output to the background estimate, using the same cut:

```bash
python burn/estimate_background.py 13 \
  analysis/results/testTree_vars_s13_BDTD.root \
  analysis/background --target_cut 0.45
```
