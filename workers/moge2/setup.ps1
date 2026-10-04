param([string]$Uv = 'C:\Users\drewy\.local\bin\uv.exe')
$ErrorActionPreference = 'Stop'
$workerRoot = $PSScriptRoot
$runtimeRoot = Join-Path $workerRoot '.runtime'
$sourceRoot = Join-Path $runtimeRoot 'upstream'
$helperRoot = Join-Path $runtimeRoot 'utils3d'
$modelRoot = Join-Path $runtimeRoot 'models/26b477f41595707c5db6770294c0d1721e8ed4ed'
$venvRoot = Join-Path $workerRoot '.venv'
if ((Test-Path -LiteralPath $sourceRoot) -or (Test-Path -LiteralPath $helperRoot) -or (Test-Path -LiteralPath $venvRoot)) {
    throw 'MoGe source/environment already exists. Inspect before rerunning setup.'
}
New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
foreach ($repo in @(
    @{Url='https://github.com/microsoft/MoGe.git'; Path=$sourceRoot; Ref='74fbce054ebed49800de42d0ad0e83495065719a'},
    @{Url='https://github.com/EasternJournalist/utils3d-moge.git'; Path=$helperRoot; Ref='62f09d58509485564e24d5d9f6aac9ee9ebc0c37'}
)) {
    git -c core.autocrlf=false clone --filter=blob:none --no-checkout $repo.Url $repo.Path
    if ($LASTEXITCODE -ne 0) { throw 'Source clone failed' }
    git -C $repo.Path config core.autocrlf false
    git -C $repo.Path config core.eol lf
    git -C $repo.Path checkout --detach $repo.Ref
    if ($LASTEXITCODE -ne 0) { throw 'Pinned source checkout failed' }
}
& $Uv venv --python 3.12 $venvRoot
if ($LASTEXITCODE -ne 0) { throw 'Isolated Python environment failed' }
$workerPython = Join-Path $venvRoot 'Scripts/python.exe'
& $Uv pip install --python $workerPython --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu'
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed' }
$inferenceRequirements = Get-Content -LiteralPath (Join-Path $workerRoot 'requirements.txt') | Where-Object { $_ -notmatch '^torch==' }
& $Uv pip install --python $workerPython --index-url https://pypi.org/simple @inferenceRequirements
if ($LASTEXITCODE -ne 0) { throw 'Pinned inference dependency installation failed' }
New-Item -ItemType Directory -Path $modelRoot -Force | Out-Null
foreach ($name in @('model.pt', 'README.md')) {
    $uri = 'https://huggingface.co/Ruicheng/moge-2-vits-normal/resolve/26b477f41595707c5db6770294c0d1721e8ed4ed/' + $name
    Invoke-WebRequest -UseBasicParsing -Uri $uri -OutFile (Join-Path $modelRoot $name)
}
$restoreGitBytes = @'
import hashlib, subprocess, sys
from pathlib import Path
for root in sys.argv[1:]:
    source = Path(root)
    for row in subprocess.check_output(['git', '-C', root, 'ls-tree', '-r', 'HEAD'], text=True).splitlines():
        meta, name = row.split('\t', 1)
        path = source / name
        data = path.read_bytes()
        if hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() != meta.split()[2]:
            path.write_bytes(subprocess.check_output(['git', '-C', root, 'show', f'HEAD:{name}']))
'@
$restoreGitBytes | & $workerPython - $sourceRoot $helperRoot
if ($LASTEXITCODE -ne 0) { throw 'Exact Git source byte restoration failed' }

& $workerPython (Join-Path $workerRoot 'worker.py') --identity
if ($LASTEXITCODE -ne 0) { throw 'Installed worker identity validation failed' }
& $Uv pip freeze --python $workerPython | Set-Content -LiteralPath (Join-Path $runtimeRoot 'installed-packages.txt') -Encoding UTF8
