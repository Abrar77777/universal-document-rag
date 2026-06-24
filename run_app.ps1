$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Find-SystemPython {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return @($pythonCmd.Source)
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        return @($pyCmd.Source, "-3")
    }

    throw "Python was not found. Install Python 3.10+ or add it to PATH, then rerun .\run_app.ps1"
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment..."
    $SystemPython = Find-SystemPython
    if ($SystemPython.Count -gt 1) {
        & $SystemPython[0] $SystemPython[1] -m venv .venv
    }
    else {
        & $SystemPython[0] -m venv .venv
    }
}

Write-Host "Checking dependencies..."
$DependencyCheck = & $VenvPython -c "import importlib.util; missing=[p for p in ['fastapi','streamlit','pandas','faiss','openai'] if importlib.util.find_spec(p) is None]; print('missing:' + ','.join(missing) if missing else 'ok')" 2>$null
if ($LASTEXITCODE -ne 0 -or $DependencyCheck -notcontains "ok") {
    Write-Host "Installing dependencies. This can take a few minutes the first time..."
    & $VenvPython -m pip install -r requirements.txt.txt
}

Write-Host "Starting presentation demo..."
& $VenvPython scripts\run_demo.py
