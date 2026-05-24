$ErrorActionPreference = "Stop"

$root = Resolve-Path "$PSScriptRoot\.."
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$env:PYTHONPATH = Join-Path $root "src"

& $python -m qsr.train_router_v7 `
  --config (Join-Path $root "configs\router_v7_train.json")
