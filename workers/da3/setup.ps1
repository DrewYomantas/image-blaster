param([string]$Uv = 'C:\Users\drewy\.local\bin\uv.exe')
$ErrorActionPreference = 'Stop'
$workerRoot = $PSScriptRoot
$runtimeRoot = Join-Path $workerRoot '.runtime'
$sourceRoot = Join-Path $runtimeRoot 'upstream'
$modelRoot = Join-Path $runtimeRoot 'models/e08cab65ca0ec38e7826075418411ab90cab4da3'
$venvRoot = Join-Path $workerRoot '.venv'

if (Test-Path -LiteralPath $sourceRoot) {
    throw 'Upstream source already exists. Inspect it before rerunning setup.'
}
New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
git -c core.autocrlf=false clone --filter=blob:none --no-checkout https://github.com/ByteDance-Seed/Depth-Anything-3.git $sourceRoot
if ($LASTEXITCODE -ne 0) { throw 'Source clone failed' }
git -C $sourceRoot -c core.autocrlf=false checkout --detach 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
if ($LASTEXITCODE -ne 0) { throw 'Pinned source checkout failed' }
& $Uv venv --python 3.12 $venvRoot
if ($LASTEXITCODE -ne 0) { throw 'Isolated Python environment failed' }
$workerPython = Join-Path $venvRoot 'Scripts/python.exe'
& $Uv pip install --python $workerPython --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu' 'torchvision==0.22.1+cpu'
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed' }
$inferenceRequirements = Get-Content -LiteralPath (Join-Path $workerRoot 'requirements.txt') | Where-Object { $_ -notmatch '^torch(?:vision)?==' }
& $Uv pip install --python $workerPython --index-url https://pypi.org/simple @inferenceRequirements
if ($LASTEXITCODE -ne 0) { throw 'Pinned inference dependencies installation failed' }
New-Item -ItemType Directory -Path $modelRoot -Force | Out-Null
foreach ($name in @('config.json', 'model.safetensors', 'README.md')) {
    $uri = 'https://huggingface.co/depth-anything/DA3-SMALL/resolve/e08cab65ca0ec38e7826075418411ab90cab4da3/' + $name
    Invoke-WebRequest -UseBasicParsing -Uri $uri -OutFile (Join-Path $modelRoot $name)
}
& $workerPython (Join-Path $workerRoot 'worker.py') --identity
if ($LASTEXITCODE -ne 0) { throw 'Installed worker identity validation failed' }
& $Uv pip freeze --python $workerPython | Set-Content -LiteralPath (Join-Path $runtimeRoot 'installed-packages.txt') -Encoding UTF8
