# Q-SR

This repository contains the public code and lightweight run scripts for
Q-SR, a selective-recovery pipeline for studying safety routing and
over-refusal reduction in quantized small language models.

Only code, configuration templates, and execution wrappers are included here.
The manuscript, figures, raw benchmark data, completions, checkpoints, audit
packets, provider outputs, and result ledgers are intentionally not part of
this repository.

## Layout

- `src/qsr/`: core routers, feature extraction, recovery gating, verifier
  wrappers, training utilities, and evaluation drivers.
- `configs/`: relative-path JSON configs for local training and evaluation
  runs.
- `scripts/`: small PowerShell wrappers around the Python entry points.
- `requirements.txt`: minimal Python dependency list.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CUDA-specific packages may need to be installed according to your local GPU
and driver stack.

## Example Commands

Train local routing and calibration components after preparing your own
non-evaluation training data:

```powershell
.\scripts\train_router_v6.ps1
.\scripts\train_router_v7.ps1
.\scripts\train_utility_calibrator_v6.ps1
.\scripts\train_recovery_veto.ps1
```

Run the clean XSTest-style evaluation driver after placing local datasets and
model artifacts at the paths referenced by the configs:

```powershell
.\scripts\run_qsr_clean_xstest_full.ps1
```

Run the output-verified recovery scan once the required local generation
ledgers and verifier dependencies are available:

```powershell
.\scripts\run_qsr_output_verified_len160.ps1
```

## Not Included

The following are excluded on purpose:

- paper source, PDFs, figures, and supplementary material;
- benchmark datasets and raw prompts/completions;
- model checkpoints, adapters, and calibrator artifacts;
- human-audit packets, annotator notes, and private keys;
- provider probes, raw judge outputs, logs, and large generated artifacts.

Place local datasets under `data/` and trained artifacts under `checkpoints/`
when running experiments. These paths are ignored by Git.
