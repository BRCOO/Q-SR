$ErrorActionPreference = "Stop"

$root = Resolve-Path "$PSScriptRoot\.."
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$env:PYTHONPATH = Join-Path $root "src"

& $python -m qsr.train_recovery_veto `
  --config (Join-Path $root "configs\recovery_veto_train.json")
