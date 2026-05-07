# ReDisCA Examples

This folder contains runnable workflows that show the library on article-style
synthetic data, ready MNE evoked data, and ERP CORE N170 data.

Run these commands from the repository root with the project environment:

```bash
.venv/bin/python -m pip install -e ".[dev,mne]"
```

If your virtual environment is already activated, `python` is fine. Plain
system `python3` may fail with `ModuleNotFoundError: No module named 'mne'`.

## Article-Style Synthetic Data

```bash
.venv/bin/python examples/synthetic_article.py
```

This is the only synthetic example. It runs compact simulations inspired by the
article:

- one representational source mixed into sensors;
- several active sources, each with its own RDM.

There are no command-line arguments. Edit the user settings at the top of
`examples/synthetic_article.py`:

```python
OUTPUT_ROOT = ROOT / "examples" / "repro_outputs" / "synthetic_article"
N_ITER = 30
SNR_LEVELS = [0.35, 0.70, 1.05]
N_CHANNELS = 32
N_TIMEPOINTS = 200
SMOOTH_SIGMA = 10.0
RDM_NOISE_SCALE = 0.25
RANDOM_STATE = 0
```

Outputs go to:

```text
examples/repro_outputs/synthetic_article/
```

The script saves:

- `benchmark_overview.png`: metric distributions over SNR levels
- `single_source_example.png`: true RDM, noisy target RDM, recovered RDM,
  and true/recovered sensor pattern for one single-source run
- `multi_source_example.png`: true, target, and recovered RDMs for each
  planted source, plus a true/recovered sensor-pattern comparison
- `single_source_trials.csv` and `multi_source_trials.csv`: raw Monte Carlo
  metrics
- `summary.json`: compact config and summary metrics

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
- saves ICA diagnostic figures for manual inspection
- applies selected ICA exclusions
- epochs, baseline-corrects, and averages the four conditions
- writes the compact ReDisCA bundle under `examples/n170/prepared/`

The download step requires internet access and uses the ERP CORE N170 files
from OSF. MNE is still used for EEG processing and visualization, but not for
fetching this N170 dataset. If the files were downloaded manually, place them
here before running the preparation script:

```text
examples/n170/data/erpcore_n170/sub-001/eeg/
```

Downloaded data and generated artifacts are ignored by git:

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

To manually change rejected ICA components, inspect the figures, edit
`MANUAL_ICA_EXCLUDE` in `examples/n170/prepare_erpcore_n170.py`, and rerun the
preparation script.

`examples/n170/reproduce_erpcore_n170.py` is a plain Python script. Edit the user
settings at the top of the file:

```python
READY_NPZ = N170_ROOT / "prepared" / "erpcore_n170_sub001_ready.npz"
READY_INFO_FIF = N170_ROOT / "prepared" / "erpcore_n170_sub001_info.fif"
OUTPUT_ROOT = N170_ROOT / "outputs"

PERMUTATION_TEST = True
N_PERM = 1000
ALPHA = 0.05

MEANINGFUL_WINDOW_MS = 150.0
MEANINGFUL_STEP_MS = 25.0
CATEGORY_WINDOW_START_S = 0.150
CATEGORY_WINDOW_STOP_S = 0.250
```

```bash
.venv/bin/python examples/n170/reproduce_erpcore_n170.py
```

The script starts from a prepared bundle and does not load raw EEG, filter,
epoch, or average trials. The ready files are:

```text
examples/n170/prepared/erpcore_n170_sub001_ready.npz
examples/n170/prepared/erpcore_n170_sub001_info.fif
examples/n170/prepared/erpcore_n170_sub001_ready_metadata.json
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
- an article-comparison summary that separates high RDM correlation from
  permutation-test significance

Outputs go to:

```text
examples/n170/outputs/
```

## Already Prepared Data

`examples/analyze_ready_data.py` is a plain Python script for the case where
preprocessing is already done. Edit the user settings at the top of the file:

```python
INPUT_NPZ = ROOT / "examples" / "data" / "prepared_data.npz"
OUTPUT_ROOT = ROOT / "examples" / "repro_outputs" / "ready_data"

TMIN = 0.150
TMAX = 0.250

PERMUTATION_TEST = True
N_PERM = 1000

RUN_SLIDING_WINDOW = True
WINDOW_MS = 150.0
STEP_MS = 25.0
```

The input `.npz` file should contain:

- `X`: condition-averaged data with shape `(C, N, T)`
- `target_rdm`: target RDM with shape `(C, C)`
- `times`: time axis in seconds with shape `(T,)`
- `condition_order`: optional condition names
- `sfreq`: optional sampling frequency, required when `RUN_SLIDING_WINDOW = True`

```bash
.venv/bin/python examples/analyze_ready_data.py
```

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
