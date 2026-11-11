@echo off
setlocal
:: Brings the Python dependencies up to date when requirements.txt changed.
:: Called by RunPhotoProcessing.bat right after it pulls the latest code:
::     call UpdateDeps.bat "<path to the env's python.exe>"
::
:: "Changed" means the committed requirements.txt differs from the one last
:: installed successfully, remembered as its git blob hash in a stamp file
:: next to the env's python.exe (so a failed install is retried next time).
:: Only the pip packages are refreshed: PyTorch and dlib are hardware-specific
:: (CUDA builds) and stay as InitPhotoProcessing.bat installed them. Never
:: stops the launcher: a problem here only prints a warning.

set "APP_DIR=%~dp0."
set "ENV_PY=%~1"
if not defined ENV_PY exit /b 0
if not exist "%APP_DIR%\.git" exit /b 0

set "NEW_BLOB="
for /f %%H in ('git -C "%APP_DIR%" rev-parse HEAD:requirements.txt 2^>nul') do set "NEW_BLOB=%%H"
if not defined NEW_BLOB exit /b 0

for %%I in ("%ENV_PY%") do set "ENV_DIR=%%~dpI"
set "STAMP=%ENV_DIR%.requirements-blob"
set "DONE_BLOB="
if exist "%STAMP%" set /p DONE_BLOB=<"%STAMP%"
if "%NEW_BLOB%"=="%DONE_BLOB%" exit /b 0

echo requirements.txt changed - updating dependencies...
set "REQ_FILTERED=%TEMP%\bpd-requirements.filtered.txt"
:: dlib comes from conda (see InitPhotoProcessing.bat), not pip.
findstr /V /R "^dlib" "%APP_DIR%\requirements.txt" > "%REQ_FILTERED%"
"%ENV_PY%" -m pip install -r "%REQ_FILTERED%"
if errorlevel 1 goto :warn
:: Same pins InitPhotoProcessing.bat re-asserts after installing requirements.
"%ENV_PY%" -m pip install "numpy>=1.26.0,<2" "setuptools<81"
if errorlevel 1 goto :warn

>"%STAMP%" echo %NEW_BLOB%
echo Dependencies are up to date.
exit /b 0

:warn
echo WARNING: could not update the dependencies - starting anyway. It will be retried next time.
exit /b 0
