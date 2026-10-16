param(
    [switch]$NoAudio,
    [string]$Root = "",
    [string]$BundleRevision = "f59b9962e77ed16dc5687282e5108b1f9958640c"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
if (-not [Environment]::Is64BitOperatingSystem) {
    throw "This check needs 64-bit Windows."
}
if ($env:AEC_BUNDLE_REVISION) { $BundleRevision = $env:AEC_BUNDLE_REVISION }
if ($BundleRevision -notmatch '^[0-9a-f]{40}$') { throw "Missing pinned test revision." }
if (-not $Root) {
    $Root = Join-Path $env:LOCALAPPDATA 's2s-aec'
}
$Root = [IO.Path]::GetFullPath($Root)
$run = Join-Path $Root ("results\" + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6))
New-Item -ItemType Directory -Force $run | Out-Null
Start-Transcript -Path (Join-Path $run 'setup.log') | Out-Null
$exitCode = 0
try {
    Write-Host "Setting up the Windows audio check. No models, account or server needed."
    Write-Host "Test files and Python stay inside: $Root"
    $uvVersion = '0.11.28'
    $uvDir = Join-Path $Root "uv-$uvVersion"
    $uv = Join-Path $uvDir 'uv.exe'
    if (-not (Test-Path $uv)) {
        $zip = Join-Path $run 'uv.zip'
        Invoke-WebRequest -UseBasicParsing "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $uvDir -Force
        Remove-Item $zip
    }
    $source = Join-Path $Root ("src-" + $BundleRevision.Substring(0, 12))
    if (-not (Test-Path $source)) {
        $zip = Join-Path $run 'source.zip'
        Invoke-WebRequest -UseBasicParsing "https://github.com/jackboyla/speech-to-speech/archive/$BundleRevision.zip" -OutFile $zip
        $unpack = Join-Path $Root ('unpack-' + [guid]::NewGuid().ToString('N').Substring(0, 6))
        Expand-Archive -Path $zip -DestinationPath $unpack
        Move-Item (Join-Path $unpack "speech-to-speech-$BundleRevision") $source
        Remove-Item $zip
        Remove-Item $unpack
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Root 'python'
    $env:UV_CACHE_DIR = Join-Path $Root 'cache'
    $env:UV_LINK_MODE = 'copy'
    $env:PYTHONPATH = Join-Path $source 'src'
    $env:PYTHONUNBUFFERED = '1'
    $env:AEC_BUNDLE_REVISION = $BundleRevision
    $venv = Join-Path $Root 'client-env'
    & $uv venv --python 3.11 --managed-python $venv --allow-existing
    if ($LASTEXITCODE -ne 0) { throw 'Python setup failed; see setup.log.' }
    $python = Join-Path $venv 'Scripts\python.exe'
    & $uv pip install --python $python --only-binary :all: `
        'numpy==2.4.3' 'openai==2.28.0' 'jsonschema==4.26.0' `
        'pywebrtc-audio==0.2.0' 'sounddevice==0.5.3' 'soundfile==0.14.0' 'soxr==1.1.0' `
        'scipy==1.17.1' 'pytest==9.1.1' 'pytest-asyncio==1.4.0'
    if ($LASTEXITCODE -ne 0) { throw 'Dependency setup failed; see setup.log.' }

    Write-Host "Running the PR's existing audio-client tests on Windows..."
    & $python -m pytest --noconftest -o asyncio_mode=auto -q `
        (Join-Path $source 'tests\openai_realtime\test_audio_client.py') `
        "--junitxml=$(Join-Path $run 'tests.xml')"
    if ($LASTEXITCODE -ne 0) { throw 'Audio-client regression checks failed. Results saved in tests.xml.' }

    $audioArgs = @('--results', $run)
    if ($NoAudio) { $audioArgs += '--no-audio' }
    & $python (Join-Path $source 'scripts\windows_aec_check.py') @audioArgs
    if ($LASTEXITCODE -ne 0) { throw 'The audio check could not finish. See report.json and setup.log.' }
} catch {
    $exitCode = 1
    Write-Host "CHECK STOPPED: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Stop-Transcript | Out-Null
    Write-Host "Results: $run"
    $resultZip = $run + '.zip'
    Compress-Archive -Path (Join-Path $run '*') -DestinationPath $resultZip
    if (-not $NoAudio) {
        Start-Process explorer.exe -ArgumentList ('/select,"' + $resultZip + '"')
    }
}
if ($exitCode -ne 0) { throw "Windows check failed. Keep the results folder for diagnosis: $run" }
