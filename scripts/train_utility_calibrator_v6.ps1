$ErrorActionPreference = "Stop"

$root = Resolve-Path "$PSScriptRoot\.."
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$env:PYTHONPATH = Join-Path $root "src"

& $python -m qsr.train_utility_calibrator `
  --config (Join-Path $root "configs\utility_calibrator_v6_train.json")
