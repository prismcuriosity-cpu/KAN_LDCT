# Full run on a single RTX 5090 (32 GB, ~30% held in reserve).
# Usage:  .\scripts\run_all.ps1 -DataRoot "D:\datasets\ct-low-dose-reconstruction\Preprocessed_256x256\256"

param(
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [string]$OutDir = "runs/kan_corediff_256",
    [int]$ImgSize = 256,
    [int]$BatchSize = 16,
    [int]$EpochsPretrain = 30,
    [int]$EpochsPhysics = 5,
    [int]$EpochsBaseline = 30
)

$ErrorActionPreference = "Stop"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "== 0/4  self-check ==" -ForegroundColor Cyan
python tests\test_smoke.py
if (-not $?) { throw "smoke test failed - fix before training" }

$common = @(
    "--data_root", $DataRoot, "--out_dir", $OutDir,
    "--img_size", $ImgSize, "--batch_size", $BatchSize,
    "--epochs_pretrain", $EpochsPretrain,
    "--epochs_physics", $EpochsPhysics,
    "--epochs_baseline", $EpochsBaseline
)

Write-Host "== 1/4  restoration pre-training ==" -ForegroundColor Cyan
python -m kanldct.train --stage pretrain @common

Write-Host "== 2/4  physics / KAN heads ==" -ForegroundColor Cyan
python -m kanldct.train --stage physics @common

Write-Host "== 3/4  baselines ==" -ForegroundColor Cyan
python -m kanldct.train --stage baselines @common

Write-Host "== 4/4  held-out evaluation ==" -ForegroundColor Cyan
python -m kanldct.evaluate @common

Write-Host "results in $OutDir" -ForegroundColor Green
