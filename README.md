# Can Mamba match LSTM for daily streamflow prediction?

Reference implementation for the manuscript *"Can Mamba match LSTM for daily
streamflow prediction? A parameter- and depth-matched benchmark on 671 CAMELS
basins"*.

The paper trains a Mamba-3 **Static-Dynamic Adapter** and a CuDNN LSTM in
capacity-matched pairs (nine pairs, 18 runs) on 671 CAMELS basins and reports
that the LSTM leads in median NSE and KGE in every pair at a fixed 30-epoch
endpoint. This repository contains the training, evaluation, statistical, and
figure code behind those numbers.

---

## What is in the repository

| Path | Role |
|---|---|
| `config.py` | Every tunable knob: tier ladders, recipes, seeds, output paths |
| `run_all_tiers.py` | Entry point for the 18-run benchmark and the diagnostics |
| `training.py` | Training loop, loss, optimizer and scheduler construction |
| `data_utils.py` | CAMELS download, scaling, basin normalisation, loaders |
| `CudnnLstmModel.py` | Multi-layer CuDNN LSTM baseline |
| `mamba_model.py` | Mamba-3 blocks plus the Static-Dynamic Adapter |
| `evaluation.py` | Per-basin NSE, KGE components, FHV, FLV, peak metrics |
| `sign_test_lstm_vs_mamba.py` | Blocked cluster permutation test, headline pair |
| `sign_test_all_pairs.py` | The same test extended to all nine matched pairs |
| `ecdf_percentile_analysis.py` | ECDF crossing analysis and self-checks |
| `plot_figures.py` | Regenerates the manuscript figures and their data CSVs |
| `plot_camels_conus_distribution.py` | CONUS hydroclimatic maps |
| `count_model_parameters.py` | Verifies the parameter counts behind the matching |
| `dump_gauge_ids.py` | Writes `gauge_ids.csv` from CAMELS if absent |
| `generate_mc_dropout.py` | Stochastic forward passes for MC-dropout analysis |
| `extract_table_data.py` | Dumps table values from the run outputs |
| `results/` | Two small derived CSVs, shipped so the statistics are checkable |
| `tests/` | Metric smoke tests |

### Inputs kept in the repository

- `camels_attributes_v2.0/`: CAMELS climate and topography attribute tables.
- `gauge_ids.csv`: the 671 basin identifiers, names, and coordinates.
- `us_states.geojson`: boundary geometry for the map figures.

The time series themselves are **not** included. `data_utils.py` downloads
them on first use from the CAMELS distribution.

### Run outputs are not in this repository

The four experiment directories total roughly 3.7 GB, which is far beyond what
a code repository should hold. They are archived separately; see
**Data and outputs** below.

---

## Setup

### The submitted results were produced on WSL

Every number in the manuscript was generated **inside the Windows Subsystem for
Linux (WSL 2, Ubuntu)**, not in native Windows. This is not incidental:
`mamba-ssm` and `causal-conv1d` compile CUDA extensions at install time and are
the practical reason to work in a Linux environment. On native Windows the Mamba
path is likely to fail to build.

The exact stack that produced the submitted results:

| Component | Version |
|---|---|
| WSL | 2, Ubuntu |
| OS layer | Python 3.12.13 (conda env `py312`) |
| PyTorch | 2.13.0+cu130 |
| CUDA driver | 610.47 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB |
| `mamba-ssm` | 2.3.2.post1 |
| `causal-conv1d` | 1.6.2.post1 |
| `hydroDL` | 0.1.6 (editable checkout) |
| numpy / pandas / scipy | 2.5.1 / 3.0.5 / 1.18.0 |

### Setting it up under WSL

From a Windows PowerShell prompt, in your project directory:

```powershell
wsl --install -d Ubuntu            # once, if WSL is not installed yet
```

Then, inside WSL:

```bash
cd /mnt/c/Users/your-windows-username/your-project-directory
# then clone this repository and enter its directory

conda create -n py312 python=3.12 -y
conda activate py312

pip install -r requirements.txt
```

The repository sits on the Windows filesystem via `/mnt/...`, which is how it
was run. Be aware that training I/O is noticeably faster if the checkout lives
inside the Linux filesystem (`~/...`) rather than on `/mnt/...`; that was not
done here, so the timings below include the `/mnt` overhead.

### Setting it up natively (not recommended)

```bash
python -m venv .venv
source .venv/Scripts/activate       # Windows

pip install -r requirements.txt
```

The LSTM half of the code, and the metric tests, work this way. The Mamba half
generally will not, because of the CUDA extension builds above.

**`mamba-ssm` and `causal-conv1d` build CUDA extensions at install time** and
are the most common source of failure. Install them from a wheel matching your
CUDA version, or build with `MAMBA_FORCE_BUILD=TRUE pip install mamba-ssm`.
The Mamba-3 path will not run without them; nothing else requires them.

Verify the environment before doing anything expensive:

```bash
python tests/test_metrics.py
```

This pins the metric code to hand-checkable values (perfect prediction, uniform
scaling, constant offset, NaN and zero-variance handling, KGE recomposition)
and exits non-zero on any failure. It needs no GPU and no run outputs. If it
cannot import `evaluation.py` because `hydroDL` is missing, it reports a skip
rather than a false pass.

It was last run in the environment tabulated above, where all 7 test groups
pass:

```
7 test group(s) passed, 0 failed
```

Note that the tests need `hydroDL` importable. On a machine without it the script
exits 0 having printed `SKIP`, which is a skip, not a pass. Check for the word
`SKIP` rather than trusting the exit code alone.

---

## Reproducing the paper

### 1. The 18-run benchmark

One command produces every configuration reported in the main results table:

```bash
python run_all_tiers.py
```

That is three capacity tiers, three depths, both architectures, run in isolated
subprocesses so GPU memory is fully reclaimed between runs. On the hardware
reported in the paper (a single RTX 4060 Laptop GPU) the eighteen runs total
about 295 minutes, of which 150 are the nine LSTM runs and 145 the nine Adapter
runs, so expect roughly five hours end to end.

The parameter matching is enforced, not assumed: every Adapter is paired with an
LSTM of the same depth and within 0.50% of the same parameter count. Confirm it
with:

```bash
python count_model_parameters.py
```

Smaller pieces, useful before committing to a full run:

```bash
python run_all_tiers.py --tiers 1 --epoch 1 --model mamba --force   # smoke test
python run_all_tiers.py --list-runs                                  # completed runs
python run_all_tiers.py --tiers 1 --model mamba --eval-only          # rescore only
```

### 2. Data and outputs

Code: <https://github.com/shriar/Mamba_vs_LSTM_Streamflow_prediction>

Download the archived run outputs from Zenodo and unpack them into the
repository root:

```bash
curl -L -o Output.zip \
  https://zenodo.org/api/records/23087719/files/Output.zip/content
unzip Output.zip -d .
rm Output.zip
```

That archive is 1.86 GB compressed (3.61 GB uncompressed), MD5
`31ba2deda844a41701a0309aa17a1fcc`.

The analysis scripts look for four directories in the repository root, by these
exact names:

```
output 7 -- full run -- LSTM and Mamba
output 8 --- LSTM and mamba run with alternate hyperparamter
output 9 --- rho 1095
output 10 --- epoch 60
```

They contain spaces and runs of dashes because the Python modules reference
them literally. Renaming them breaks the analysis pipeline. They map to the
paper as follows:

| Directory | Paper content |
|---|---|
| `output 7` | The 18-run matched benchmark, all headline results |
| `output 8` | Two-way optimizer-recipe swap control |
| `output 9` | The $\rho = 1095$ lookback diagnostic |
| `output 10` | The 60-epoch extended-training diagnostic |

### 3. Statistics

```bash
python sign_test_lstm_vs_mamba.py    # headline pair
python sign_test_all_pairs.py        # all nine matched pairs
```

Both use a cluster-level blocked permutation test. Whole spatial blocks are
relabelled jointly rather than basins individually, because neighbouring CAMELS
basins share climate, soils, and a single regional model; treating them as 671
independent draws understates the null. The test runs at block sizes of 3, 4,
and 5 degrees over 200,000 permutations and reports every one, because the
result depends on that choice and quoting a single block size would be
arbitrary. Implementation notes worth knowing:

- The null is built by **joint** flips, not within-block permutation. Permuting
  labels inside a block leaves that block's win count unchanged, so the global
  win share is invariant and the null degenerates to $p = 1$. Independent
  per-basin flips are the opposite error: they reduce to the plain sign test
  and ignore the blocks entirely.
- The p-value tests *which model wins each basin*, not the size of the gap.
- Two CSVs are already committed under `results/` so the reported statistics can
  be read without running anything.

### 4. Figures

```bash
python plot_figures.py                 # all figures
python plot_figures.py --figures 9     # just one
python plot_figures.py --list-runs
python plot_figures.py --output-root "path/to/output 7 -- full run -- LSTM and Mamba"
```

Each figure writes both a PNG and a `*_data.csv` holding the plotted values, so
every number in the manuscript can be traced to a file. Figures are written to
`figures - Results/` by default; pass `--fig-dir` to change it.

### Which table or figure comes from where

| Manuscript item | Command | Source file |
|---|---|---|
| Main benchmark table (18 rows) | `run_all_tiers.py` | `output 7/*/All/basin_metrics.csv` |
| KGE components, FHV, FLV | `run_all_tiers.py` | `output 7/*/All/kge_components.csv` |
| Blocked permutation $p$-values | `sign_test_all_pairs.py` | `results/paired_pvalues_all_pairs.csv` |
| Per-basin ECDF percentiles | `ecdf_percentile_analysis.py` | ECDF and quantile CSVs |
| Recipe-swap table | `output 8` | `basin_metrics.csv` in that directory |
| Lookback table | `output 9` | `basin_metrics.csv` in that directory |
| Learning curves, 60-epoch runs | `output 10` | `checkpoint_metrics.csv` |
| ECDF, scaling, training-time figures | `plot_figures.py` | `figures - Results/*_data.csv` |
| Peak-event diagnostics | `plot_figures.py` | `fig11_peak_events_data.csv` |
| Seasonal heatmaps | `plot_figures.py` | `fig09_seasonal_heatmaps_data.csv` |

### Architecture variants

`run_all_tiers.py --mamba-variant` accepts `adapter` (reported in the paper),
`film`, `hybrid`, `prefix`, and `residual`. Only `adapter` is used for any
number in the manuscript; the others are retained because they were used during
development.

---

## Scope and honest limitations

These are stated here as well as in the paper, because a reader with the code
in hand can see them directly.

- **Single seed.** Every run in the primary matrix uses seed `111111`. There is
  no ensemble and no across-run variability estimate, so the results are point
  estimates, not replicated effects. The blocked permutation test establishes
  that the *ordering* is not chance, given spatial clustering; it says nothing
  about seed-to-seed spread.
- **Fixed endpoint.** All 18 runs are scored at epoch 30. The Adapter's test
  metrics peak earlier, so the epoch choice affects the size of the gap and, on
  KGE, briefly its sign. `tab:epoch_robustness` in the paper grants both models
  their most favourable epoch and reports the outcome either way.
- **Implementation dependency.** The training, loss, and epoch definitions come
  from `hydroDL`. The exact revision used for the submitted runs was not pinned,
  so small implementation details (the loss denominator, recurrent dropout
  placement, the number of optimiser updates per epoch) should be read as
  describing the code as it stood, not as a specification independent of
  `hydroDL`.
- **Optimiser control is partial.** The two-way swap changes learning rate,
  cosine floor, weight decay, and clipping together, and covers one
  configuration pair. It bounds the ranking against an optimiser artefact; it
  does not isolate which setting matters.
- **The comparison is between two complete designs.** The LSTM receives static
  attributes before recurrence; the Adapter injects them after each block. The
  experiment therefore compares model designs, not recurrence types in
  isolation.

---

## Data

CAMELS (Newman et al., 2015; Addor et al., 2017) is distributed by NCAR at
<https://ral.ucar.edu/solutions/products/camels> and is downloaded
automatically on first use. It is not redistributed here.

Daymet forcings are used as distributed with CAMELS.

---

## License

MIT. See `LICENSE`.
