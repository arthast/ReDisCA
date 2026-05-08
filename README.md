# ReDisCA

ReDisCA is a Python implementation of Representational Dissimilarity Component
Analysis for EEG/MEG-style evoked responses.

The library finds spatial components whose condition-to-condition
dissimilarity structure matches a user-defined target RDM.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

For tests:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

For MNE topomaps and EEG/MEG example scripts:

```bash
.venv/bin/python -m pip install -e ".[mne]"
```

For the optional comparison example against `mne-rsa`:

```bash
.venv/bin/python -m pip install -e ".[mne,mne-rsa]"
```

All example commands below assume the same project environment. If `python3`
does not see MNE on your machine, use `.venv/bin/python`.

## Data Shape

The core API expects averaged evoked responses:

```text
X.shape = (C, N, T)
```

where:

- `C` is the number of conditions
- `N` is the number of sensors/channels
- `T` is the number of time points

For MNE workflows, the high-level helpers can accept condition-averaged
`Evoked` objects directly, so user scripts do not need to manually construct
this tensor.

## Quick Start

```python
import numpy as np

from redisca import binary_rdm, export_result, fit_redisca
from redisca.viz import plot_component_timeseries, plot_top_component_rdms

condition_order = ["face", "car", "scrambled_face", "scrambled_car"]
target_rdm = binary_rdm(
    condition_order,
    positive_conditions={"face", "car"},
)

X = np.random.randn(4, 16, 200)

result = fit_redisca(
    X,
    target_rdm,
    permutation_test=True,
    n_perm=500,
    random_state=0,
)

plot_top_component_rdms(result, k=3)
plot_component_timeseries(result, idxs=[0], condition_names=condition_order)
export_result(result, "redisca_output")

print(result.pearson_scores)
print(result.p_values)
```

Permutation testing reshuffles the upper-triangular target-RDM entries against
the fixed condition-pair data matrices and reports component-wise p-values.

## Sliding Windows

Use sample counts when you already work in samples:

```python
from redisca import sliding_window_fit_redisca

scan = sliding_window_fit_redisca(
    X,
    target_rdm,
    window_size=50,
    step_size=10,
)
```

Use milliseconds when working with EEG/MEG sampling frequency:

```python
from redisca import sliding_window_fit_redisca_ms

scan = sliding_window_fit_redisca_ms(
    X,
    target_rdm,
    sfreq=1024.0,
    window_ms=150.0,
    step_ms=25.0,
    times=times,
    permutation_test=True,
    n_perm=1000,
)

pearson_over_time = scan.component_metric_matrix("pearson_scores")
p_values_over_time = scan.component_metric_matrix("p_values")
```

## Visualization

Lightweight Matplotlib plots live in `redisca.viz`:

- `plot_rdm`
- `plot_top_component_rdms`
- `plot_component_scores`
- `plot_component_lambdas`
- `plot_sliding_window_metric`
- `plot_component_timeseries`
- `plot_patterns`

MNE-aware plots live in `redisca.viz_mne`:

- `plot_pattern_topomaps`
- `plot_compare_conditions`
- `plot_condition_joint`

Batch report helpers live in `redisca.report`:

- `save_evoked_overview`
- `save_scan_overview_figure`
- `save_window_sequence_figure`
- `save_component_summary_figure`
- `save_result_diagnostics`
- `save_sliding_window_report`
- `save_window_metrics_csv`

Example:

```python
from redisca.report import save_scan_overview_figure

save_scan_overview_figure(
    scan,
    "figures/scan_overview.png",
    info=info,
    condition_names=condition_order,
    title="Sliding-window summary",
)
```

## Examples

See [examples/README.md](examples/README.md) for a short guide.

Main scripts:

- `examples/synthetic_benchmark.py` runs synthetic benchmark simulations.
- `examples/analyze_mne_sample_evokeds.py` runs ReDisCA on ready MNE sample evokeds.
- `examples/compare_mne_rsa_sample_evokeds.py` compares ReDisCA with sensor-level MNE-RSA.
- `examples/n170/prepare_erpcore_n170.py` downloads/prepares ERP CORE N170 with ICA diagnostics.
- `examples/n170/reproduce_erpcore_n170.py` runs ERP CORE N170 analysis from a prepared bundle.

Prepared ERP CORE N170 data are expected under:

```text
examples/n170/prepared/
```

Generated figures and reports are written to example-specific output folders,
such as `examples/n170/outputs/` and `examples/repro_outputs/`.
