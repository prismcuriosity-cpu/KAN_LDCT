# Ablation grid.  Each run writes its own results_test.txt; collect with
#   Get-ChildItem runs\abl_* -Filter results_test.txt -Recurse | %{ $_.FullName; Get-Content $_ }
param(
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [int]$Epochs = 15
)
$ErrorActionPreference = "Stop"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$runs = @(
    @{ name = "full";        args = @() },
    @{ name = "no_kan_unet"; args = @("--use_kan_unet", "false") },
    @{ name = "no_kan_time"; args = @("--use_kan_time", "false") },
    @{ name = "bspline";     args = @("--kan_impl", "bspline") },
    @{ name = "no_twostage"; args = @("--two_stage", "false") },
    @{ name = "no_dc";       args = @("--dc_step", "0") },
    @{ name = "no_kan_acc";  args = @("--use_kan_acc", "false") },
    @{ name = "T1";          args = @("--n_steps", "1") },
    @{ name = "T20";         args = @("--n_steps", "20") }
)

foreach ($r in $runs) {
    Write-Host "== ablation: $($r.name) ==" -ForegroundColor Cyan
    $out = "runs/abl_$($r.name)"
    python -m kanldct.train --stage pretrain --data_root $DataRoot --out_dir $out `
        --epochs_pretrain $Epochs @($r.args)
    python -m kanldct.train --stage physics  --data_root $DataRoot --out_dir $out @($r.args)
    python -m kanldct.evaluate --data_root $DataRoot --out_dir $out @($r.args)
}
