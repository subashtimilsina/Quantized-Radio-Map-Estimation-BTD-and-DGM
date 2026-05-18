# Quantized Radio Map Estimation (BTD + DGM)

<p align="center">
  <img src="https://github.com/XiaoFuLab/Quantized-Radio-Map-Estimation-BTD-and-DGM/blob/master/demo.png?raw=true" alt="Quantized spectrum cartography demo" width="720"/>
</p>

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/10335642"><b>Paper (IEEE Xplore)</b></a>
</p>

Official PyTorch code accompanying:

> **Quantized Radio Map Estimation Using Tensor and Deep Generative Models**  
> Subash Timilsina, Sagar Shrestha, Xiao Fu — *IEEE Transactions on Signal Processing*, 2024.

The project estimates **spatial loss field (SLF) maps** from **coarsely quantized** RF measurements. It implements:

- **BTD** — block-term decomposition (tensor factorization) prior on SLFs.
- **DGM** — deep generative model (GAN) prior, with latent optimization.

Both use a **probit-based quantized observation model** and alternating optimization over coefficients and latent factors.

---

## Table of contents

- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Data and pretrained models](#data-and-pretrained-models)
- [Running experiments](#running-experiments)
- [Method sketch](#method-sketch)
- [Citation](#citation)

---

## Repository layout

| Path | Role |
|------|------|
| [`demo.ipynb`](demo.ipynb) | End-to-end demo: load a radio map, simulate quantization, run **BTD** or **DGM** reconstruction. |
| [`configs.py`](configs.py) | Sampling masks, quantization bin edges, learning rates, regularization. |
| [`quantization_model_log.py`](quantization_model_log.py) | Quantization in **log-domain**, probit likelihood `p(Y \| X̂)`, tensor CP-style products, NMSE metric. |
| [`CostFunction.py`](CostFunction.py) | Negative log-likelihood + Frobenius regularization for BTD vs DGM. |
| [`Trans.py`](Trans.py) | Log transform and inverse (`TransformLog`) applied before/after the likelihood. |
| [`qmc_utils.py`](qmc_utils.py) | Load `.mat` experiments, optional plotting, GAN checkpoint loading (`Generator256`). |
| [`gan.py`](gan.py) | **Inference** GAN architecture (`Generator256`, `Discriminator`) for 51×51 maps. |
| [`train_utils.py`](train_utils.py) | **Training** GAN (`Generator` / `Discriminator` for 14×34 maps), MAT dataset class `SLFDataset`. |
| [`train_gan.py`](train_gan.py) | Script to train the smaller GAN on MAT files under `Data_generation/Real_data_train/`. |
| [`quantization_bin_discovery.py`](quantization_bin_discovery.py) | Optional utilities to summarize quantiles across `.pt` tensors and aggregate bins from CSV. |

---

## Setup

**Python** 3.8+ recommended.

```bash
cd Quantized-Radio-Map-Estimation-BTD-and-DGM
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# If needed, reinstall torch with the CUDA build from https://pytorch.org/
```

---

## Data and pretrained models

### Simulating radio maps

You can **simulate** synthetic spatial loss fields and related quantities with the companion repository [**Radio-map-simulator**](https://github.com/XiaoFuLab/Radio-map-simulator) (same paper citation; see its `README` for `generate_radio_map.py`, `run.sh`, and CLI options such as emitters, grid size, and bandwidth length). Use its outputs as a starting point if you need fresh scenarios beyond the `.mat` layouts expected here.

The demo expects:

1. **Radio map MAT file** — e.g. `RadioMaps/2_2.mat` with variables such as `S`, `T`, `C`, `S_true`, `C_true`, `T_true` (see `load_data` in `qmc_utils.py` for layout and permutations).
2. **Pretrained GAN checkpoint** — e.g. `Model/sngan11_256_unnorm.pt` with `g_model_state_dict` (see `load_generator` in `qmc_utils.py`).

> **Note:** Large assets are excluded by [`.gitignore`](.gitignore) (`*.mat`, `*.pt`, …). Add your own data and model under the paths used in `demo.ipynb`, or change those paths to match your layout.

To **train a GAN** from MAT tiles, prepare:

```
Data_generation/Real_data_train/slf_mat/
  000001.mat
  000002.mat
  ...
```

Each file should contain a `Sc` variable (tensor read in `SLFDataset`). Then run `python train_gan.py`. Logs go to `logs/<timestamp>/`; weights to `Models/GAN/`.

---

## Running experiments

### Quick demo (notebook)

```bash
jupyter lab demo.ipynb
# or: jupyter notebook demo.ipynb
```

Typical flow inside the notebook:

1. Set `GAN_PATH` and `DATA_PATH` to your checkpoint and `.mat` scenario.
2. Choose `SELECT_METHOD` as `"btd"` or `"dgm"` in `optimize_for(...)`.
3. Adjust `sampling_percent` (fraction of observed pixels) and optional `sigma_val` (probit noise scale).

### GAN training (script)

```bash
python train_gan.py
```

Uses GPU when available (`cuda`). Training hyperparameters are at the bottom of `train_gan.py` (epochs, batch size, learning rate, latent dim).

---

## Method sketch

1. **Forward model** — True SLF tensor `T` is built from emitters and coefficients; a log transform `h_η` is applied before quantization.
2. **Observation** — `Y = Q(h_η(T) + E)` with Gaussian `E` and fixed bin boundaries; `Y` stores **bin indices**.
3. **Inverse problem** — Maximize probit likelihood of `Y` under estimate `T̂`, with a **sampling mask** `W` (partial observations).
4. **Prior** — Either **BTD** (low-rank slice factors in `Z`) or **DGM** (`T̂` from a generator on latent `Z`).
5. **Optimization** — Alternating Adam steps on coefficients `C` and latent `Z` (with non-negativity projections where used).

See the paper for full problem statement and identifiability discussion.

---

## Citation

If you use this code, please cite:

```bibtex
@article{timilsina2023quantized,
  title={Quantized radio map estimation using tensor and deep generative models},
  author={Timilsina, Subash and Shrestha, Sagar and Fu, Xiao},
  journal={IEEE Transactions on Signal Processing},
  volume={72},
  pages={173--189},
  year={2023},
  publisher={IEEE}
}
```
