# ReDisCA Examples

This folder contains runnable workflows that show the library on synthetic
benchmark data, ready MNE evoked data, and ERP CORE N170 data.

Run these commands from the repository root with the project environment:

```bash
.venv/bin/python -m pip install -e ".[dev,mne]"
```

If your virtual environment is already activated, `python` is fine. Plain
system `python3` may fail with `ModuleNotFoundError: No module named 'mne'`.

## Synthetic Benchmark

```bash
.venv/bin/python examples/synthetic_benchmark.py
```

This is the only synthetic example. It runs a multi-source simulation:

- 4 planted sources;
- 6 conditions;
- source time series generated as `S = M @ Z`;
- noisy target RDMs and 1/f-like sensor noise.

Outputs go to:

```text
examples/repro_outputs/synthetic_benchmark/
```

The script saves:

- `source_recovery_metrics.png`: recovery metrics over SNR levels
- `source_recovery_example.png`: true, target, and recovered RDMs for each
  planted source, plus a true/recovered sensor-pattern comparison
- `source_recovery_trials.csv`: raw Monte Carlo metrics

## Ready MNE Evokeds

```bash
.venv/bin/python examples/analyze_mne_sample_evokeds.py
```

This is the simplest real-data example. It uses MNE's built-in sample dataset
and starts from ready averaged `Evoked` responses in `sample_audvis-ave.fif`.
No raw-data preprocessing is performed by this script.

The script picks four conditions:

- `Left Auditory`
- `Right Auditory`
- `Left visual`
- `Right visual`

It builds a target RDM for auditory-vs-visual responses, runs:

- a fixed-window ReDisCA analysis
- an optional sliding-window ReDisCA scan

Outputs go to:

```text
examples/repro_outputs/mne_sample_evokeds/
```

## ERP CORE N170

If the raw ERP CORE files are available and the prepared bundle should be
rebuilt, run:

```bash
.venv/bin/python examples/n170/prepare_erpcore_n170.py
```

The preparation script does the dataset-specific work once:

- downloads the ERP CORE N170 OSF files when they are missing
- loads raw ERP CORE EEG
- sets channel types, electrode montage, and average reference
- fits ICA on EEG channels
- automatically detects EOG-related ICA components
- fills ICA exclusions up to three components using the strongest EOG scores
- saves ICA diagnostic figures for manual inspection
- applies selected ICA exclusions
- epochs, baseline-corrects, and averages the four conditions
- writes the compact ReDisCA bundle under `examples/n170/prepared/`

Downloaded data and generated artifacts:

```text
examples/n170/data/
examples/n170/prepared/
examples/n170/work/
examples/n170/outputs/
```

ICA diagnostics are saved here:

```text
examples/n170/work/ica_diagnostics/
```

```bash
.venv/bin/python examples/n170/reproduce_erpcore_n170.py
```

The script starts from a prepared bundle and does not load raw EEG, filter,
epoch, or average trials. The analysis script reads:

```text
examples/n170/prepared/erpcore_n170_sub001_ready.npz
examples/n170/prepared/erpcore_n170_sub001_info.fif
```

The `.npz` file contains:

- `X`: condition-averaged responses with shape `(4, 30, 615)`
- `times`: time axis in seconds
- `condition_order`: `face`, `car`, `scrambled_face`, `scrambled_car`
- `sfreq`: sampling frequency
- `epoch_counts`: accepted epoch count per condition

Then it runs:

- a meaningful-vs-meaningless 150 ms sliding-window scan
- a fixed 150-250 ms face-specific N170 analysis
- a fixed 150-250 ms car-specific N170 analysis
- interactive Matplotlib/MNE figures for RDMs, topographies, component time
  series, and sliding-window p-values

Saved figures are grouped under:

```text
examples/n170/outputs/figures/
```

- `overview/`: condition-averaged evoked overview
- `meaningful_vs_meaningless/`: target RDM, p-values, topomap, RDMs, and
  combined sliding-window figures
- `face_specific/`: fixed-window face component figures
- `car_specific/`: fixed-window car component figures

## What Belongs In User Code

ReDisCA can provide helpers for common plumbing, but user scripts should still
state the scientific choices explicitly:

- dataset path
- preprocessing
- event-code groups
- condition order
- target RDMs
- time windows
- permutation count
