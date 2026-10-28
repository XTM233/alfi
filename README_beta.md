# Reproducing the *kr* experiments (beta)

Scratch/beta guide for reproducing the Krüppel (`kr`) spatio-temporal
transcriptomics results. Supersedes the experiment instructions in `README.md`.

Tested on: macOS arm64 (Apple Silicon), CPU only, conda env `local_alfi`.

---

## 1. What this reproduces

`pde_experiment.py` fits a **partial latent force model** (reaction–diffusion
PDE solved with legacy FEniCS) to the Drosophila gap-gene dataset:

- input: `data/clean_kr.csv` (8 time points x 64 spatial positions = 512 observations)
- outputs: protein (`prot`) and mRNA (`mrna`) predictions
- metrics: `q2`, `cia`, `smse` for both outputs

Three modelling variants are reported in the paper:

| Variant | CLI flag | Latent transform `G(u)` | Protein scaling |
|---|---|---|---|
| `0kr` | `--trans ""` | identity | `StandardScaler` |
| `1kr` | `--trans softplus_1` | `softplus(u, beta=1)` | `MinMaxScaler` |
| `2kr` | `--trans softplus_5` | `softplus(u, beta=5)` | `MinMaxScaler` |

The `0kr/1kr/2kr` names are a legacy directory prefix. The current CLI builds
directory names as `<gene>_<seed>_<timestamp><note>`, so reference runs are
named `kr_<seed>_...` with the variant encoded by `--trans`.
Do **not** pass `--gene 0kr`: the dataset only accepts `kr`, `kni`, `gt`
(`alfi/datasets/datasets.py:215-216`).

---

## 2. Environment

Two layers are required:

1. **conda** — Python 3.9 + FEniCS 2019.1 (`dolfin`) + MPI + `pkg-config`/clang.
   Legacy FEniCS is conda-forge only and provides the JIT compiler used by
   `dijitso`; it cannot be installed with pip.
2. **pip** — PyTorch/GPyTorch stack plus two git forks (`torch-fenics` and
   `barkm/pyadjoint@torch-fenics`) that provide differentiable FEniCS solves.

### 2.1 Create the environment

```bash
conda env create -f environment.yml
conda activate local_alfi

# editable install of the alfi package (setup.py is stale; runtime deps come
# from requirements.txt)
pip install -e .
```

If you prefer to install the pip layer separately:

```bash
conda create -n local_alfi -c conda-forge python=3.9 fenics=2019.1 \
      mpich mpi4py pkg-config clangxx pip
conda activate local_alfi
pip install -r requirements.txt
pip install -e .
```

### 2.2 Verify

```bash
python -c "import torch, gpytorch, dolfin, fenics_adjoint, torch_fenics; \
print(torch.__version__, gpytorch.__version__, dolfin.__version__)"
python pde_experiment.py --help
```

### 2.3 Verified versions

| Package | Version | Source |
|---|---|---|
| Python | 3.9.23 | conda-forge |
| `fenics` / `dolfin` | 2019.1.0 | conda-forge |
| `dolfin-adjoint` (`fenics_adjoint`) | 2018.1.0 | `barkm/pyadjoint` @ `torch-fenics` |
| `torch-fenics` | 0.1 | `barkm/torch-fenics` @ master |
| `torch` | 2.7.1 | PyPI |
| `gpytorch` | 1.13 | PyPI |
| `numpy` | 1.26.4 | conda-forge |
| `scipy` | 1.13.1 | PyPI |
| `pandas` | 2.3.1 | PyPI |
| `matplotlib` / `seaborn` | 3.9.4 / 0.13.2 | PyPI |
| `scikit-learn` | 1.6.1 | PyPI |
| `smt` | 2.9.3 | PyPI |
| `torchdiffeq` / `torchcubicspline` | 0.2.5 / 0.0.3 | PyPI |

Notes:

- **CPU only.** The code calls `.cuda()` only when CUDA is present; a single
  run takes roughly 2 hours on CPU.
- Always `conda activate local_alfi`; invoking the env's `python` directly was
  observed to trigger a duplicate `libomp` crash on macOS. If that happens:
  `export KMP_DUPLICATE_LIB_OK=TRUE`.
- The first PDE solve is slow while FEniCS JIT-compiles forms
  (cache: `~/.cache/dijitso`).
- `arviz` (MCMC plotters) and TensorFlow (legacy experiments) are **not**
  needed for these runs.

---

## 3. Run the experiments

Run everything **from the repository root** — the dataset path is relative
(`data_dir='./data'`, `pde_experiment.py:229-232`).

### 3.1 Exact command for the completed batch

The reported batch (`experiment_results/kr_42..45_*final-no-zero-mean-sample-10-lr-0.005`)
was produced by `run.sh` (seeds 42-45, two at a time). The full single-run
command, with every hyperparameter written out as an argument, is:

```bash
python pde_experiment.py --gene kr --SEED 42 --drosophila True --use_lhs True --train_ratio 0.3 --skip False --learn_inducing_locations False --gp_use_tril True --params_grad False --experiment_dir experiment_results --note "final-no-zero-mean-sample-10-lr-0.005" --max_epoch 300 --debug False --lr 0.005 --trans ""
```

Run it once per seed (`42`, `43`, `44`, `45`). `--trans ""` selects the
identity (0kr) variant. Booleans accept `True`/`False` (`_str_to_bool` in
`pde_experiment.py`), so `--skip False` keeps the 80-epoch pre-estimator.

Each run writes one timestamped directory:

```
experiment_results/kr_<SEED>_<timestamp>final-no-zero-mean-sample-10-lr-0.005/
├── result.json                  # all-data metrics + held-out *_test metrics + n_train/n_test
├── early_res.json               # best training-time metrics + timings
├── 300epochs_trained_lfm.pt     # trained LFM checkpoint
├── 300epochs_trained_gp.pt      # trained GP checkpoint
├── loss.pdf                     # loss / ELBO / KL curves
├── prot.pdf, mrna.pdf           # spatio-temporal predictions
├── prot_unscaled.pdf            # protein plot in data units
└── training_times.npy           # per-epoch time, loss, ELBO, KL
```

The checkpoint prefix is `<max_epoch>epochs_trained_`, so `--max_epoch 300`
produces `300epochs_trained_*`. See section 4 for the `result.json` key layout.

Parameters fixed in code (`pde_experiment.py:253-273`): `lengthscale=10`,
`natural=True`, `zero_mean=False`, `warm_epochs=-1`, `clamp=True`,
`parameter_grad=False`. The PDE parameters (`sensitivity`, `decay`,
`diffusion`) come from `dros_ground_truth('kr')` and stay fixed unless
`--params_grad True`. The gene default of 100 epochs
(`pde_experiment.py:253`) is overridden by `--max_epoch 300`.

Batch:

```bash
./run.sh
```

### 3.2 What the seed controls

The seed is the replicate index (paper error bars use seeds 42–45). It sets:

1. global RNGs (`pde_experiment.py:214-219`) → parameter/variational init and
   all Monte-Carlo draws (`rsample`, `alfi/models/partial_lfm.py:61`);
2. inducing-point locations via `LHS(..., random_state=RandomState(SEED))`
   (`pde_experiment.py:323`);
3. the 30% training mask via `randperm(..., generator=manual_seed(SEED))`
   (`pde_experiment.py:394`) — about 153 of 512 observations;
4. the 5 posterior samples used for final metrics (`pde_experiment.py:537`).

It does **not** change the data, PDE parameters, epochs or learning rate.
The headline `result.json` metrics are computed over the whole dataset; the
held-out complement of the training mask is also scored under the `*_test`
keys (section 4), so seed variance in the headline numbers partly reflects
which points happened to be trained on.

---

## 4. Outputs

Each run writes to `experiment_results/<gene>_<seed>_<timestamp><note>/`:

| File | Contents |
|---|---|
| `result.json` | final all-data metrics, held-out `*_test` metrics, `n_train`/`n_test` |
| `early_res.json` | best metrics seen during training + timings |
| `300epochs_trained_lfm.pt` / `_gp.pt` | model checkpoints (`<num_epochs>` prefix) |
| `loss.pdf` | loss, ELBO and KL trajectories |
| `prot.pdf`, `mrna.pdf`, `prot_unscaled.pdf` | spatio-temporal predictions |
| `training_times.npy` | per-epoch time, loss, ELBO, KL |

`result.json` is the metric to report. Unsuffixed keys (`prot_q2`, `mrna_cia`, …)
are computed on **all 512** observations; `*_test` keys are computed on the
**held-out complement** of `train_mask` (359 points for `--train_ratio 0.3`).
`n_train`/`n_test` record the split sizes; when `n_test == 0`
(e.g. `--train_ratio 1.0`) the `*_test` values are `null`. `early_res.json`
records the best training-time values (all-data) and may look better. Runs made
before the held-out metrics were added do not contain the `*_test`/`n_*` keys.

---

## 5. Aggregating results

`process_results.py` builds `experiment_stats_temp.csv` with mean ± sd per
variant. Three caveats:

- it only scans directories starting with `0kr`, `1kr`, `2kr`
  (`process_results.py:163-175`), so the current `kr_*` runs are ignored —
  rename the prefix or adapt the filter;
- if you reuse the new `result.json` schema in those directories, filter the
  `n_train`/`n_test`/`*_test` keys out of the aggregation (the current script
  averages every key it finds);
- the `2kr` inverse transform is commented out (`process_results.py:369-372`),
  so that plotting path is unfinished.

Reference aggregates from the Aug 2025 batch:

| Variant | prot_q2 | mrna_q2 | prot_cia (%) | mrna_cia (%) |
|---|---|---|---|---|
| 0kr | 0.8425 (0.0526) | 0.7686 (0.0146) | 36.63 (6.26) | 8.78 (3.98) |
| 1kr | 0.8884 (0.0106) | 0.8163 (0.0610) | 6.63 (18.87) | -22.55 (5.19) |
| 2kr | 0.8853 (0.0128) | 0.7940 (0.0373) | -1.22 (4.93) | -20.83 (2.68) |

(From `experiment_stats_temp.csv`; `cia` is reported as `(0.6827 - mean) * 100`.)

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'dolfin'` | install the conda layer from `environment.yml` |
| `OSError: pkg-config probably not installed` | activate `local_alfi`; `conda install -c conda-forge pkg-config` |
| `OMP: Error #15 ... libomp.dylib already initialized` | activate the env; fallback `export KMP_DUPLICATE_LIB_OK=TRUE` |
| `AssertionError` while loading data | `--gene` must be `kr`, `kni` or `gt` |
| `FileNotFoundError: data/clean_kr.csv` | run from the repo root |
| Very slow first solve | FEniCS JIT compile; warm-up populates `~/.cache/dijitso` |
| No GPU used | expected: runs are CPU-bound and CPU-only here |

---

## 7. Known limitations

- `setup.py` is stale (no FEniCS/smt/scipy pins); `requirements.txt` is now the
  source of truth. Docs deps moved to `requirements-docs.txt`.
- `run.sh` hard-codes the conda env name `local_alfi`.
- `pde_experiment.py:602` has a `FIXME` about the protein/mRNA plot naming.
- The headline metrics are all-data; held-out `*_test` metrics are now saved in
  `result.json`, but `process_results.py` does not aggregate them yet and runs
  from before the change do not contain them.
- Seed results are reproducible on the same hardware/versions, not bitwise
  across platforms.
