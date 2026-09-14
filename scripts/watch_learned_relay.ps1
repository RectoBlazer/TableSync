param([int]$Seed = 0)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv-train\Scripts\python.exe'
$EvaluatorPath = Join-Path $PSScriptRoot 'evaluate_act.py'
$CheckpointPath = Join-Path $ProjectRoot 'outputs\act_residual\checkpoints\step_000250'
$RunName = 'hybrid_view_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
$OutputPath = Join-Path (Join-Path $ProjectRoot 'artifacts') $RunName

Write-Host 'TableSync: training trajectory reference + learned ACT corrections.'
Write-Host 'Both arms relay the block through the table. Close the viewer to stop.'
& $PythonPath $EvaluatorPath --checkpoint $CheckpointPath --seeds $Seed --output $OutputPath --view
exit $LASTEXITCODE
