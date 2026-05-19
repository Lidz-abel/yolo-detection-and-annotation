$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$AppName = "YOLO_Detection_Tool"
$DistDir = Join-Path $Root "dist"
$AppDir = Join-Path $DistDir $AppName
$ExePath = Join-Path $AppDir "$AppName.exe"

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)]
    [string] $FilePath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
  )
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
  }
}

Write-Host "Checking Python version..."
$PythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python version: $PythonVersion"
if ($PythonVersion -eq "3.13") {
  Write-Host "Warning: Python 3.13 may be incompatible with some PyInstaller/PyTorch hooks. Python 3.10 or 3.11 is recommended."
}

Write-Host "Installing/upgrading build tools..."
Invoke-Checked python -m pip install --upgrade pip setuptools wheel pyinstaller pyinstaller-hooks-contrib

Write-Host "Building YOLO desktop executable..."
Invoke-Checked python -m PyInstaller `
  --noconfirm `
  --clean `
  --onedir `
  --windowed `
  --name $AppName `
  "desktop_app\yolo_desktop_app_v2.py"

if (-not (Test-Path $ExePath)) {
  throw "PyInstaller finished but exe was not found: $ExePath"
}

Write-Host "Copying model, config, metadata, and training-verification files..."
Copy-Item -Recurse -Force "configs" $AppDir
Copy-Item -Recurse -Force "exports" $AppDir
Copy-Item -Recurse -Force "outputs" $AppDir
Copy-Item -Recurse -Force "metadata" $AppDir
Copy-Item -Recurse -Force "models" $AppDir
Copy-Item -Recurse -Force "losses" $AppDir
Copy-Item -Recurse -Force "utils" $AppDir
Copy-Item -Recurse -Force "backend" $AppDir

Write-Host ""
Write-Host "Build complete."
Write-Host "Double-click this file to start:"
Write-Host $ExePath
