# Cleanup script for model files
$modelDir = "models/pretrained"

# Get size before cleanup
$sizeBefore = (Get-ChildItem -Path $modelDir -Recurse | Measure-Object -Property Length -Sum).Sum
Write-Host "Size before cleanup: $($sizeBefore / 1GB) GB"

# Files to keep
$filesToKeep = @(
    "best_model.pt"  # Keep the best model
)

# Find the latest pd_model_epoch100 file (final model)
$latestEpoch100 = Get-ChildItem -Path $modelDir -Filter "pd_model_epoch100_*.pt" | 
                 Sort-Object LastWriteTime -Descending | 
                 Select-Object -First 1
if ($latestEpoch100) {
    $filesToKeep += $latestEpoch100.Name
    Write-Host "Keeping latest epoch 100 model: $($latestEpoch100.Name)"
}

# Delete files instead of moving to backup
foreach ($file in Get-ChildItem -Path $modelDir -Filter "*.pt") {
    if (-not ($filesToKeep -contains $file.Name)) {
        Write-Host "Deleting $($file.Name)"
        Remove-Item -Path "$modelDir/$($file.Name)" -Force
    } else {
        Write-Host "Keeping $($file.Name)"
    }
}

# Also delete the backup directory if it exists
if (Test-Path "models/backup") {
    Write-Host "Deleting backup directory"
    Remove-Item -Path "models/backup" -Recurse -Force
}

# Get size after cleanup
$sizeAfter = (Get-ChildItem -Path $modelDir -Recurse | Measure-Object -Property Length -Sum).Sum
Write-Host "Size after cleanup: $($sizeAfter / 1GB) GB"
Write-Host "Reduced by: $(($sizeBefore - $sizeAfter) / 1GB) GB" 