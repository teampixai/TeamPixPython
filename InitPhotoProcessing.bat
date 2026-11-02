@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: InitPhotoProcessing.bat
:: ----------------------------------------------------------------------------
:: One-shot / re-runnable bootstrapper for TeamPixPython (Windows).
::
:: Fully self-contained: everything the app needs at runtime - its own private
:: Miniconda, Python, every dependency, and the app source - lives under the
:: install root. A system-wide Anaconda/Miniconda/Python (already installed,
:: installed later, or changed later) can't break the app, and installing the
:: app doesn't modify the system-wide one. Nothing needs to be pre-installed
:: (no conda, no git); designed to be downloaded and run on its own from a
:: plain Command Prompt.
::
:: What it does:
::   1. Detects the CPU architecture and, if this install root has no private
::      Miniconda yet, downloads the latest official Miniconda installer from
::      repo.anaconda.com, checks its Authenticode signature, asks you to
::      accept Anaconda's Terms of Service (skip the prompt with /y), and
::      installs it silently into <install_root>\miniconda3 WITHOUT adding it
::      to PATH, registering it as the system Python, writing to the registry,
::      or creating its own shortcuts. Windows on ARM64 uses the x86_64 build
::      (Anaconda publishes no native win-arm64 Miniconda; it runs under
::      Windows' x64 emulation). Any system-wide conda is never used.
::      For the rest of this script (and in the generated launcher) conda's
::      config / package cache / env registration are redirected into the
::      install root, and inherited Python/pip/conda settings (PYTHONPATH,
::      user site-packages, pip.ini, an activated system conda env, ...) are
::      cleared so they can't leak in.
::   2. Creates an isolated conda environment (Python 3.12) with the private
::      conda, and makes sure pip AND git are installed in it (a plain
::      `conda create ... python=X.Y` doesn't always pull pip in, and git is
::      always installed into the env rather than relying on a system Git -
::      git isn't used for `git clone`/`git+` URLs anywhere below, but some
::      pip source installs, e.g. setuptools_scm version detection, shell out
::      to it). An environment left behind by an older version of this script
::      (built with a system-wide conda) is rebuilt once with the private one.
::   3. Pins setuptools<81 right away so pkg_resources stays available for
::      face_recognition_models - done before the much slower steps below so
::      a broken pip environment is caught fast.
::   4. Detects an NVIDIA GPU/driver (nvidia-smi, driver >= 528 for CUDA 12) to
::      pick CUDA vs CPU builds.
::   5. Downloads the latest TeamPixPython source as a GitHub ZIP archive
::      (no git required) and updates the local app folder in place (existing
::      albums\ and other local-only data are never deleted).
::   6. Installs dlib via conda-forge (prebuilt binary - no C++ compiler /
::      CMake needed) - GPU build + cudnn when an NVIDIA GPU was detected,
::      otherwise the CPU build.
::   7. Installs PyTorch 2.2.2 (CUDA 12.1 build or CPU-only; pinned because
::      facenet-pytorch requires torch<2.3), then checks afterwards that
::      PyTorch and dlib can really use the GPU (warns if not).
::   8. Installs the rest of requirements.txt via pip.
::   9. Installs face_recognition_models straight from source (PyPI releases
::      of it are unreliable) via a GitHub ZIP archive URL (no git required).
::  10. Writes a RunPhotoProcessing.bat launcher that runs the app with the
::      private environment, isolated the same way as this script.
::  11. Creates a per-user Start Menu shortcut ("TeamPixPython") that
::      starts the launcher (no admin rights needed).
::
:: Safe to re-run at any time: it reuses the private Miniconda, re-checks /
:: updates the environment and dependencies, pulls the latest app code, and
:: refreshes the launcher and shortcut.
::
:: Usage:
::   InitPhotoProcessing.bat [/y] [install_root]
::     /y            accept Anaconda's Terms of Service without prompting when
::                   the private Miniconda is installed (also -y or
::                   --accept-license). Use only if you have read them:
::                   https://www.anaconda.com/legal
::     install_root  optional override for the directory the private
::                   Miniconda, conda environment, app source, and launcher
::                   are placed in (default: the current directory, i.e. the
::                   folder this script is run from). A path without spaces
::                   is recommended (Miniconda advises against spaces).
:: ============================================================================

:: Clear settings inherited from the user's environment - e.g. a system-wide
:: Anaconda activated via `conda init` / cmd AutoRun, a PYTHONPATH pointing at
:: another Python's packages, or pip/conda options that redirect installs
:: (PIP_INDEX_URL, PIP_EXTRA_INDEX_URL, CONDA_CHANNELS, CONDA_SUBDIR, ...).
:: Every CONDA_*/_CE_*/PIP_* variable is dropped; the ones we need are set
:: further down. Must stay before this script sets any CONDA_* variable of
:: its own (e.g. CONDA_DIR), or those would be wiped too.
for %%P in (CONDA_ _CE_ PIP_) do (
    for /f "delims==" %%V in ('set %%P 2^>nul') do set "%%V="
)
for %%V in (VIRTUAL_ENV PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE) do set "%%V="

set "REPO_OWNER=teampixai"
set "REPO_NAME=TeamPixPython"
set "REPO_BRANCH=main"
set "ENV_NAME=teampixpython"
set "PYTHON_VERSION=3.12"
:: PyTorch versions are pinned because requirements.txt's facenet-pytorch
:: requires torch<2.3 / torchvision<0.18; installing a newer (e.g. CUDA)
:: torch first would just get replaced by a CPU-only 2.2.x when
:: requirements.txt is installed. The newest CUDA 12 index that publishes
:: torch 2.2.2 wheels is cu121, which needs an NVIDIA driver >= 528 on Windows.
set "TORCH_SPEC=torch==2.2.2 torchvision==0.17.2"
set "TORCH_CUDA_TAG=cu121"
set "MIN_CUDA_DRIVER=528"

set "INSTALL_ROOT=%CD%"
set "ACCEPT_LICENSE=0"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="/y" goto :arg_accept
if /i "%~1"=="-y" goto :arg_accept
if /i "%~1"=="--accept-license" goto :arg_accept
set "INSTALL_ROOT=%~1"
shift
goto :parse_args
:arg_accept
set "ACCEPT_LICENSE=1"
shift
goto :parse_args
:args_done

:: The Miniconda installer's /D= switch needs an absolute path.
for %%I in ("%INSTALL_ROOT%") do set "INSTALL_ROOT=%%~fI"
if "%INSTALL_ROOT:~-1%"=="\" set "INSTALL_ROOT=%INSTALL_ROOT:~0,-1%"

echo ============================================================
echo  TeamPixPython - environment setup
echo  Install root: %INSTALL_ROOT%
echo ============================================================
echo.

if not exist "%INSTALL_ROOT%" mkdir "%INSTALL_ROOT%"

set "APP_DIR=%INSTALL_ROOT%\app"
set "WORK_DIR=%INSTALL_ROOT%\_setup_tmp"
set "CONDA_DIR=%INSTALL_ROOT%\miniconda3"
set "ENV_DIR=%INSTALL_ROOT%\envs\%ENV_NAME%"
set "ENV_PY=%ENV_DIR%\python.exe"
:: Marks an env created by the private Miniconda (see step 2).
set "ENV_MARKER=%ENV_DIR%\.bpd-private-conda"

if not exist "%WORK_DIR%" mkdir "%WORK_DIR%"

:: ----------------------------------------------------------------------------
:: Isolate this script from any other Python / conda / pip on the machine
:: (inherited variables were already cleared at the top of this script).
:: ----------------------------------------------------------------------------
:: Ignore the per-user site-packages folder and every pip.ini/pip.conf.
set "PYTHONNOUSERSITE=1"
set "PIP_CONFIG_FILE=NUL"
:: Point conda at the private Miniconda so nothing it spawns (post-link
:: scripts, `conda` called by name) can resolve to a system-wide conda.
set "CONDA_EXE=%CONDA_DIR%\Scripts\conda.exe"
set "CONDA_PYTHON_EXE=%CONDA_DIR%\python.exe"
:: Keep conda's config, package cache, and env list inside the install root.
:: Conda still also reads a user-level %USERPROFILE%\.condarc, so every conda
:: install below passes --override-channels with explicit channels.
:: register_envs=false keeps our env out of %USERPROFILE%\.conda\environments.txt
:: (so a system-wide conda won't list or manage it).
set "CONDARC=%INSTALL_ROOT%\condarc"
set "CONDA_PKGS_DIRS=%INSTALL_ROOT%\pkgs"
set "CONDA_ENVS_DIRS=%INSTALL_ROOT%\envs"
set "CONDA_REGISTER_ENVS=false"
set "CONDA_AUTO_UPDATE_CONDA=false"
:: Don't let packages create their own Start Menu entries (e.g. conda-forge's
:: git adds "Git Bash (<env>)") - the only shortcut is the one from step 11.
set "CONDA_SHORTCUTS=false"
(
    echo # Private conda config for TeamPixPython - generated by InitPhotoProcessing.bat.
    echo # Only used by the private Miniconda in this folder ^(via the CONDARC variable^).
    echo channels:
    echo   - defaults
    echo register_envs: false
    echo auto_update_conda: false
    echo shortcuts: false
) > "%CONDARC%"

:: ----------------------------------------------------------------------------
:: [1/11] Private Miniconda (downloaded + installed into the install root)
:: ----------------------------------------------------------------------------
echo [1/11] Checking for the app's private Miniconda...
set "HOST_ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "HOST_ARCH=%PROCESSOR_ARCHITEW6432%"
if /i "%HOST_ARCH%"=="AMD64" (
    set "MC_ARCH=x86_64"
) else if /i "%HOST_ARCH%"=="ARM64" (
    set "MC_ARCH=x86_64"
    echo       Windows on ARM64 detected. Anaconda publishes no native ARM64 Miniconda for
    echo       Windows, so the x86_64 build is used - it runs under Windows' x64 emulation.
) else (
    echo ERROR: unsupported CPU architecture "%HOST_ARCH%" - 64-bit Windows ^(x64 or ARM64^) is required.
    goto :fail
)

if exist "%CONDA_DIR%\Scripts\conda.exe" (
    echo       Using the private Miniconda at "%CONDA_DIR%".
    goto :conda_ready
)

echo.
echo       This app runs on its own private copy of Miniconda ^(by Anaconda, Inc.^),
echo       which provides Python and the conda package manager. The latest version
echo       will be downloaded from repo.anaconda.com and installed ONLY into:
echo         %CONDA_DIR%
echo       It is not added to PATH or the registry, and does not affect any other
echo       Anaconda, Miniconda or Python installation on this PC.
echo.
echo       Miniconda and Anaconda's package repository are subject to Anaconda's
echo       Terms of Service: https://www.anaconda.com/legal
echo.
if "%ACCEPT_LICENSE%"=="1" (
    echo       /y given - Anaconda's Terms of Service accepted on the command line.
) else (
    choice /C YN /M "      Do you accept Anaconda's Terms of Service and want to install Miniconda here"
    if errorlevel 2 goto :declined
)

if not "%CONDA_DIR: =%"=="%CONDA_DIR%" (
    echo       WARNING: the install path contains spaces. Miniconda advises against this and
    echo       some packages may fail to install. A path without spaces is recommended.
)

set "MC_FILE=Miniconda3-latest-Windows-%MC_ARCH%.exe"
set "MC_URL=https://repo.anaconda.com/miniconda/%MC_FILE%"
set "MC_EXE=%WORK_DIR%\%MC_FILE%"
if exist "%MC_EXE%" del /f /q "%MC_EXE%"
echo       Downloading %MC_URL% ...
call :download "%MC_URL%" "%MC_EXE%"
if not exist "%MC_EXE%" (
    echo ERROR: failed to download the Miniconda installer.
    goto :fail
)

echo       Verifying the installer's digital signature...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = Get-AuthenticodeSignature -LiteralPath $env:MC_EXE; if ($s.Status -ne 'Valid' -or $s.SignerCertificate.Subject -notmatch 'Anaconda') { Write-Host ('      Signature status: ' + $s.Status + ', signer: ' + $s.SignerCertificate.Subject); exit 1 }; Write-Host ('      Signed by: ' + $s.SignerCertificate.Subject)"
if errorlevel 1 (
    echo ERROR: the downloaded Miniconda installer is not validly signed by Anaconda - not running it.
    del /f /q "%MC_EXE%" >nul 2>nul
    goto :fail
)

:: Remove a partial install left by an interrupted earlier run.
if exist "%CONDA_DIR%" rmdir /s /q "%CONDA_DIR%"
echo       Installing Miniconda into "%CONDA_DIR%" ^(silent, this takes a few minutes^)...
:: /D= must be last and unquoted, even if the path has spaces (NSIS rule).
start "" /wait "%MC_EXE%" /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /NoShortcuts=1 /S /D=%CONDA_DIR%
if not exist "%CONDA_DIR%\Scripts\conda.exe" (
    echo ERROR: Miniconda installation failed - "%CONDA_DIR%\Scripts\conda.exe" not found.
    goto :fail
)
del /f /q "%MC_EXE%" >nul 2>nul
:: The installer registers its base folder in the user's shared
:: %USERPROFILE%\.conda\environments.txt, which makes a system-wide conda list
:: it. Remove just that line (best effort) so the private copy stays invisible.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$f = Join-Path $env:USERPROFILE '.conda\environments.txt'; if (Test-Path -LiteralPath $f) { $all = @([IO.File]::ReadAllLines($f)); $keep = @($all | Where-Object { $_.Trim().TrimEnd('\') -ne $env:CONDA_DIR }); if ($keep.Count -ne $all.Count) { [IO.File]::WriteAllLines($f, [string[]]$keep) } }" >nul 2>nul
echo       Miniconda installed.

:conda_ready
echo       Accepting Anaconda Terms of Service for the default channels (best effort)...
"%CONDA_DIR%\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>nul
"%CONDA_DIR%\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>nul
"%CONDA_DIR%\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >nul 2>nul

:: ----------------------------------------------------------------------------
:: [2/11] Conda environment
:: ----------------------------------------------------------------------------
:: The env lives under install_root\envs (not miniconda3\envs) so the app
:: folder layout stays the same as older versions of this script. An env
:: without the marker was created by a system-wide conda (older versions of
:: this script) and may hard-link into that conda's package cache - rebuild it
:: once with the private Miniconda so it no longer depends on anything outside
:: the install root.
if exist "%ENV_PY%" if not exist "%ENV_MARKER%" (
    echo [2/11] Existing environment was built by a different conda installation -
    echo        rebuilding it with the private Miniconda ^(one-time^)...
    rmdir /s /q "%ENV_DIR%"
    if exist "%ENV_DIR%" (
        echo ERROR: could not delete "%ENV_DIR%" - close the app ^(and anything else using that folder^) and re-run.
        goto :fail
    )
)

if exist "%ENV_PY%" (
    echo [2/11] Conda environment "%ENV_NAME%" already present - skipping creation.
) else (
    echo [2/11] Creating conda environment "%ENV_NAME%" ^(python %PYTHON_VERSION%^)...
    "%CONDA_DIR%\Scripts\conda.exe" create -y -p "%ENV_DIR%" --override-channels -c defaults python=%PYTHON_VERSION%
    if errorlevel 1 goto :fail
    type nul > "%ENV_MARKER%"
)

:: Make sure pip itself is present in the env - a plain `conda create ...
:: python=X.Y` does not always pull in pip as a dependency, and every
:: install step below (setuptools/torch/requirements.txt/
:: face_recognition_models) depends on it working.
echo       Verifying pip is installed in the environment...
"%ENV_PY%" -m pip --version >nul 2>nul
if errorlevel 1 (
    echo       pip not found - installing it into the environment...
    "%CONDA_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" --override-channels -c defaults pip
    if errorlevel 1 goto :fail
)

:: Always use a git installed INTO the env (never a system Git, which may be
:: missing or change) - not for `git clone`/`git+` (this script avoids
:: those), but some pip source installs shell out to git during their build
:: (e.g. setuptools_scm version detection) and fail confusingly otherwise.
echo       Verifying git is installed in the environment...
set "ENV_HAS_GIT=0"
if exist "%ENV_DIR%\Library\bin\git.exe" set "ENV_HAS_GIT=1"
if exist "%ENV_DIR%\Library\cmd\git.exe" set "ENV_HAS_GIT=1"
if "%ENV_HAS_GIT%"=="0" (
    echo       Installing git into the environment ^(conda-forge^)...
    "%CONDA_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" -c conda-forge --override-channels git
    if errorlevel 1 goto :fail
)

:: Put the conda environment's own folders first on PATH now - not just later
:: in the launcher - so the env's git/pip, and dlib/PyTorch's native DLLs
:: below, win over anything else on PATH for every subprocess pip or conda
:: spawns for the remainder of this script.
set "PATH=%ENV_DIR%\Library\bin;%ENV_DIR%\Library\cmd;%ENV_DIR%\Scripts;%ENV_DIR%;%PATH%"

set "GIT_PATH="
for /f "delims=" %%G in ('where git 2^>nul') do if not defined GIT_PATH set "GIT_PATH=%%G"
if not defined GIT_PATH (
    echo ERROR: git still not found after installing it into the environment.
    goto :fail
)
echo       Using git: !GIT_PATH!

:: ----------------------------------------------------------------------------
:: [3/11] Pin setuptools (pkg_resources compatibility for face_recognition_models)
:: ----------------------------------------------------------------------------
:: face_recognition_models' __init__.py still imports the old `pkg_resources`
:: API (from setuptools). Recent setuptools versions (81+) dropped
:: pkg_resources, which makes that import raise ModuleNotFoundError - caught
:: by face_recognition's own broad except clause and re-reported as the
:: misleading "please install face_recognition_models" message even though
:: the package IS installed. Pin setuptools<81 to keep pkg_resources
:: available. Done as early as possible (right after env creation, before
:: the much slower dlib/PyTorch/requirements.txt installs below) so a
:: broken pip/environment is caught immediately instead of after a long wait.
echo [3/11] Pinning setuptools ^(pkg_resources compatibility for face_recognition_models^)...
"%ENV_PY%" -m pip install "setuptools<81"
if errorlevel 1 goto :fail

:: ----------------------------------------------------------------------------
:: [4/11] Detect CUDA / NVIDIA GPU
:: ----------------------------------------------------------------------------
echo [4/11] Detecting NVIDIA GPU / CUDA driver...
set "HAS_NVIDIA_GPU=0"
set "TORCH_INDEX="
set "NV_DRIVER="
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    for /f "usebackq delims=" %%D in (`nvidia-smi "--query-gpu=driver_version" "--format=csv,noheader" 2^>nul`) do if not defined NV_DRIVER set "NV_DRIVER=%%D"
)
:: A broken driver makes nvidia-smi print an error sentence instead of a version.
if defined NV_DRIVER (
    echo !NV_DRIVER!| findstr /r /c:"^[0-9][0-9]*\.[0-9]" >nul
    if errorlevel 1 set "NV_DRIVER="
)
if defined NV_DRIVER (
    set "NV_MAJOR=0"
    for /f "tokens=1 delims=." %%M in ("!NV_DRIVER!") do set /a "NV_MAJOR=%%M" 2>nul
    if !NV_MAJOR! GEQ %MIN_CUDA_DRIVER% (
        echo       NVIDIA driver !NV_DRIVER! found - will install CUDA-enabled PyTorch ^(%TORCH_CUDA_TAG%^) and GPU-accelerated dlib.
        set "TORCH_INDEX=--index-url https://download.pytorch.org/whl/%TORCH_CUDA_TAG%"
        set "HAS_NVIDIA_GPU=1"
    ) else (
        echo       NVIDIA driver !NV_DRIVER! found but it is older than %MIN_CUDA_DRIVER% ^(needed for CUDA 12^) - installing CPU-only
        echo       PyTorch and dlib. Update the NVIDIA driver and re-run this script to enable the GPU.
    )
) else (
    echo       No working NVIDIA driver detected - will install CPU-only PyTorch and dlib.
)

:: ----------------------------------------------------------------------------
:: [5/11] Download latest app code (no git required)
:: ----------------------------------------------------------------------------
echo [5/11] Downloading latest source from GitHub ^(%REPO_OWNER%/%REPO_NAME%@%REPO_BRANCH%^)...
set "APP_ZIP=%WORK_DIR%\source.zip"
if exist "%APP_ZIP%" del /f /q "%APP_ZIP%"
call :download "https://github.com/%REPO_OWNER%/%REPO_NAME%/archive/refs/heads/%REPO_BRANCH%.zip" "%APP_ZIP%"
if not exist "%APP_ZIP%" (
    echo ERROR: failed to download the application source.
    goto :fail
)

set "EXTRACT_DIR=%WORK_DIR%\extracted"
if exist "%EXTRACT_DIR%" rmdir /s /q "%EXTRACT_DIR%"
mkdir "%EXTRACT_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%APP_ZIP%' -DestinationPath '%EXTRACT_DIR%' -Force"
if errorlevel 1 (
    echo ERROR: failed to extract the application source.
    goto :fail
)

set "EXTRACTED_SRC=%EXTRACT_DIR%\%REPO_NAME%-%REPO_BRANCH%"
if not exist "%EXTRACTED_SRC%" (
    echo ERROR: unexpected archive layout - "%EXTRACTED_SRC%" not found.
    goto :fail
)

if not exist "%APP_DIR%" mkdir "%APP_DIR%"
echo       Updating "%APP_DIR%" ^(existing albums\ and local data are preserved^)...
robocopy "%EXTRACTED_SRC%" "%APP_DIR%" /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
if errorlevel 8 (
    echo ERROR: robocopy failed to update the application folder.
    goto :fail
)

:: ----------------------------------------------------------------------------
:: [6/11] Hard-to-build native dependency: dlib (via conda-forge, prebuilt)
:: ----------------------------------------------------------------------------
:: Prefer the GPU build when an NVIDIA GPU was detected in step 4. conda-forge
:: ships dlib-cpu/dlib-gpu as separate packages; unlike the plain "dlib"
:: metapackage (which only requires cudnn to BUILD the CUDA variant, not to
:: run it), we explicitly install "cudnn" alongside dlib-gpu here so
:: cudnn_ops64_9.dll etc. actually exist at runtime - otherwise dlib fails
:: with "Could not locate cudnn_ops64_9.dll" / "Invalid handle. Cannot load
:: symbol cudnnCreateTensorDescriptor" (PyTorch's own bundled cuDNN from its
:: pip wheel is invisible to dlib, a separate native library). Falls back to
:: the CPU build if the GPU install fails for any reason (e.g. no compatible
:: cudnn/CUDA version available for this driver).
::
:: Notes from hands-on testing:
:: - Remove any previously installed dlib/dlib-cpu/dlib-gpu first: solving
::   with an existing (different-variant) dlib already in the env produces
::   unsatisfiable conflicts across dozens of historical dlib-gpu/cuda
::   builds and can take several minutes before failing.
:: - "--override-channels -c conda-forge" avoids also pulling in the slower
::   "defaults" channel, which otherwise makes the solve far slower.
:: - Installing dlib-gpu can pull in a newer numpy (e.g. 2.x) as a conda
::   dependency, silently corrupting the pip-installed numpy<2 (torch 2.2.2
::   needs numpy<2 - see requirements.txt comment). Step 8 below re-asserts
::   the numpy<2 pin via pip --force-reinstall to fix this deterministically
::   regardless of which branch ran here.
"%CONDA_DIR%\Scripts\conda.exe" remove -y -p "%ENV_DIR%" dlib dlib-cpu dlib-gpu --force >nul 2>nul
if "%HAS_NVIDIA_GPU%"=="1" (
    echo [6/11] Installing dlib-gpu + cudnn ^(conda-forge, CUDA-accelerated^)...
    "%CONDA_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" -c conda-forge --override-channels dlib-gpu cudnn
    if errorlevel 1 (
        echo       dlib-gpu install failed - falling back to CPU-only dlib.
        "%CONDA_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" -c conda-forge --override-channels dlib-cpu
        if errorlevel 1 goto :fail
    )
) else (
    echo [6/11] Installing dlib-cpu ^(conda-forge prebuilt binary - no compiler required^)...
    "%CONDA_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" -c conda-forge --override-channels dlib-cpu
    if errorlevel 1 goto :fail
)

:: ----------------------------------------------------------------------------
:: [7/11] PyTorch
:: ----------------------------------------------------------------------------
echo [7/11] Installing PyTorch ^(%TORCH_SPEC%^)...
"%ENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%ENV_PY%" -m pip install %TORCH_SPEC% !TORCH_INDEX!
if errorlevel 1 goto :fail

:: ----------------------------------------------------------------------------
:: [8/11] Remaining Python dependencies (dlib already handled by conda above)
:: ----------------------------------------------------------------------------
echo [8/11] Installing remaining dependencies from requirements.txt...
set "REQ_FILTERED=%WORK_DIR%\requirements.filtered.txt"
findstr /V /R "^dlib" "%APP_DIR%\requirements.txt" > "%REQ_FILTERED%"
"%ENV_PY%" -m pip install -r "%REQ_FILTERED%"
if errorlevel 1 goto :fail

echo       Re-asserting numpy^<2 pin ^(dlib-gpu's conda-forge numpy dependency can bump it to 2.x, which breaks torch's Tensor.numpy^(^) bridge^)...
"%ENV_PY%" -m pip install --force-reinstall "numpy>=1.26.0,<2"
if errorlevel 1 goto :fail

:: ----------------------------------------------------------------------------
:: [9/11] face_recognition_models (PyPI releases of this are unreliable / can
:: silently fail to register; installing straight from the source repo is
:: the fix the upstream face_recognition project itself recommends). Without
:: this, the dlib face-recognition provider fails at runtime with
:: "Please install `face_recognition_models`..." even though face-recognition
:: itself installed fine above. Installed via a plain GitHub ZIP-archive URL
:: (pip can install directly from a URL to a source archive) rather than a
:: `git+...` URL - git is still ensured earlier in step 2 as a safety net in
:: case pip's own build step shells out to it (e.g. setuptools_scm), but
:: this script itself never runs `git clone`/`git+`.
::
:: setuptools<81 was already pinned early in step 3 (so pkg_resources stays
:: available for face_recognition_models' __init__.py); re-assert it here
:: too since torch/requirements.txt installs above could have pulled in a
:: newer setuptools as a transitive dependency.
:: ----------------------------------------------------------------------------
echo [9/11] Re-asserting setuptools^<81 pin ^(pkg_resources compatibility^)...
"%ENV_PY%" -m pip install "setuptools<81"
if errorlevel 1 goto :fail

echo       Installing face_recognition_models ^(from source - PyPI releases are unreliable^)...
"%ENV_PY%" -m pip install --upgrade --force-reinstall --no-deps https://github.com/ageitgey/face_recognition_models/archive/refs/heads/master.zip
if errorlevel 1 goto :fail

echo       Verifying face_recognition_models imports correctly...
"%ENV_PY%" -c "import face_recognition_models" 2>nul
if errorlevel 1 (
    echo ERROR: face_recognition_models installed but failed to import.
    goto :fail
)

:: Make sure the GPU path really ended up with a CUDA-enabled PyTorch (a later
:: pip install can silently swap it for a CPU build) and a CUDA-enabled dlib.
:: Warnings only - e.g. a driver that needs a reboot shouldn't fail the setup.
if "%HAS_NVIDIA_GPU%"=="1" (
    echo       Verifying GPU support...
    "%ENV_PY%" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
    if errorlevel 1 (
        echo       WARNING: PyTorch cannot use the NVIDIA GPU ^(CPU-only build or driver problem^) - the app will run on the CPU.
    ) else (
        echo       PyTorch CUDA: OK
    )
    "%ENV_PY%" -c "import sys, dlib; sys.exit(0 if dlib.DLIB_USE_CUDA else 1)" >nul 2>nul
    if errorlevel 1 (
        echo       WARNING: dlib is the CPU build - face recognition will run on the CPU.
    ) else (
        echo       dlib CUDA: OK
    )
)

:: ----------------------------------------------------------------------------
:: [10/11] Convenience launcher
:: ----------------------------------------------------------------------------
echo [10/11] Writing launcher...
:: Prepend the env's native-library folders (Library\bin has cudnn/cuda DLLs,
:: e.g. cudnn_ops64_9.dll for GPU dlib; Scripts/DLLs are the usual conda-env
:: PATH entries) to PATH before invoking python.exe directly - without this,
:: Windows' DLL search can fail to locate cuDNN even though it IS installed
:: in the env, causing "Could not locate cudnn_ops64_9.dll" / "Invalid
:: handle. Cannot load symbol cudnnCreateTensorDescriptor" at runtime.
:: The rest of PATH is kept (after the env) so optional tools such as a
:: user-installed ffmpeg are still found. The same Python/pip/conda settings
:: cleared at the top of this script are cleared here too, and CONDA_PREFIX
:: points at the private env (as `conda activate` would), so the app never
:: picks up another Python's packages or an activated system conda env.
(
    echo @echo off
    echo setlocal
    echo :: Generated by InitPhotoProcessing.bat - runs the app on its private Python.
    echo set "PYTHONNOUSERSITE=1"
    echo set "PYTHONPATH="
    echo set "PYTHONHOME="
    echo set "PYTHONSTARTUP="
    echo set "PYTHONUSERBASE="
    echo set "VIRTUAL_ENV="
    echo set "PIP_CONFIG_FILE=NUL"
    echo set "CONDA_DEFAULT_ENV="
    echo set "CONDA_SHLVL="
    echo set "CONDA_EXE="
    echo set "CONDA_PYTHON_EXE="
    echo set "CONDA_PREFIX=%ENV_DIR%"
    echo set "PATH=%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%ENV_DIR%;%%PATH%%"
    echo "%ENV_PY%" "%APP_DIR%\culling_app.py" %%*
) > "%INSTALL_ROOT%\RunPhotoProcessing.bat"

:: ----------------------------------------------------------------------------
:: [11/11] Start Menu shortcut (per user - no admin rights needed)
:: ----------------------------------------------------------------------------
:: Created through the WScript.Shell COM object (the same kind of .lnk the
:: Anaconda installer makes). The console starts minimized - the app opens
:: its own browser tab - and the working folder matches double-clicking
:: RunPhotoProcessing.bat. Re-runs overwrite it so its paths stay current.
:: Paths are passed via environment variables to avoid quoting issues.
echo [11/11] Creating Start Menu shortcut...
set "BPD_LNK_TARGET=%INSTALL_ROOT%\RunPhotoProcessing.bat"
set "BPD_LNK_WORKDIR=%INSTALL_ROOT%"
set "BPD_LNK_ICON=%ENV_PY%"
set "BPD_LNK_PATH="
for /f "usebackq delims=" %%L in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$dir = Join-Path ([Environment]::GetFolderPath('Programs')) 'TeamPixPython'; $null = New-Item -ItemType Directory -Force -Path $dir; $lnk = Join-Path $dir 'TeamPixPython.lnk'; $s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk); $s.TargetPath = $env:BPD_LNK_TARGET; $s.WorkingDirectory = $env:BPD_LNK_WORKDIR; $s.IconLocation = $env:BPD_LNK_ICON + ',0'; $s.WindowStyle = 7; $s.Description = 'TeamPixPython photo culling web app'; $s.Save(); $lnk"`) do set "BPD_LNK_PATH=%%L"
if defined BPD_LNK_PATH (
    echo       Created "!BPD_LNK_PATH!"
) else (
    echo       WARNING: could not create the Start Menu shortcut - run RunPhotoProcessing.bat directly instead.
)

echo.
echo ============================================================
echo  Setup complete.
echo.
echo  App folder:   %APP_DIR%
echo  Python env:   %ENV_DIR%
echo  Miniconda:    %CONDA_DIR% ^(private to this app^)
echo.
echo  To start the photo culling web app, open "TeamPixPython"
echo  from the Start Menu, or run:
echo    "%INSTALL_ROOT%\RunPhotoProcessing.bat"
echo.
echo  Re-run this script any time to update everything to the latest.
echo ============================================================
pause
goto :eof

:: ----------------------------------------------------------------------------
:: :download <url> <output_path>  -- curl if available, else PowerShell.
:: ----------------------------------------------------------------------------
:download
where curl >nul 2>nul
if not errorlevel 1 (
    curl -L --fail --silent --show-error -o %2 %1
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri %1 -OutFile %2"
)
goto :eof

:declined
echo.
echo Setup cancelled - Anaconda's Terms of Service were not accepted, so Miniconda
echo was not installed. Re-run this script to try again.
pause
exit /b 1

:fail
echo.
echo Setup FAILED. See the messages above for details.
pause
exit /b 1
