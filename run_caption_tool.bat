@echo off
rem Setup virtual environment and run the Caption Sync Tool

setlocal

set "REPO_DIR=%~dp0"
set "VENV_DIR=%REPO_DIR%venv"

if not exist "%VENV_DIR%" (
    echo [+] Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

echo [+] Installing dependencies...
pip install --upgrade pip
pip install -r "%REPO_DIR%requirements.txt"

echo [+] Running Caption Tool...
python "%REPO_DIR%scripts\caption_tool.py" %*

echo.
echo [!] Script finished. Press any key to exit.
pause >nul

endlocal
