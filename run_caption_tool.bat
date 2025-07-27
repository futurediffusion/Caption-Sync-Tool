@echo off
rem Setup virtual environment and run the Caption Sync Tool

setlocal

set "REPO_DIR=%~dp0"
set "VENV_DIR=%REPO_DIR%venv"

if not exist "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

pip install -r "%REPO_DIR%requirements.txt"

python "%REPO_DIR%scripts\caption_tool.py" %*

endlocal
