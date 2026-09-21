$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Run-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program" }
}
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    Run-Checked 'python' @('-m', 'venv', '.venv')
}
Run-Checked '.venv/Scripts/python.exe' @('-m', 'pip', 'install', '-e', '.[dev]')
Run-Checked 'npm.cmd' @('--prefix', 'frontend', 'ci')
Run-Checked 'npm.cmd' @('--prefix', 'frontend', 'run', 'build')
Write-Host 'Open http://127.0.0.1:8000 in your browser.'
Run-Checked '.venv/Scripts/python.exe' @('-m', 'uvicorn', 'pathology.main:app', '--host', '127.0.0.1', '--port', '8000')
