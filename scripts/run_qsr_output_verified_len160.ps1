$ErrorActionPreference = "Stop"

$root = Resolve-Path "$PSScriptRoot\.."
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$env:PYTHONPATH = Join-Path $root "src"

& $python -m qsr.scan_v86_output_verified_recovery `
  --config (Join-Path $root "configs\qsr_output_verified_len160.json")
