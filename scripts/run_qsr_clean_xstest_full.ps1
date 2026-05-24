$ErrorActionPreference = "Stop"

$root = Resolve-Path "$PSScriptRoot\.."
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$env:PYTHONPATH = Join-Path $root "src"

& $python -m qsr.eval_qsr_v7_xstest `
  --config (Join-Path $root "configs\qsr_clean_xstest_full.json")
