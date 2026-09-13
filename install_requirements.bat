@echo off
REM ============================================================
REM  install_requirements.bat
REM  Sets up Python venv and installs all dependencies from
REM  requirements.txt. Uses relative paths so it runs on any
REM  machine without editing.
REM ============================================================

REM %~dp0 = the folder this .bat file lives in (trailing backslash included)
cd /d "%~dp0"

echo ============================================================
echo  Setting up AI Duplicate Detection Service
echo  Working directory: %cd%
echo ============================================================

REM ----- Check if Python is available -----
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on this system.
    echo Please install Python and make sure it is added to PATH.
    pause
    exit /b 1
)

REM ----- Create virtual environment if it doesn't exist -----
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] No .venv found. Creating a new virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [INFO] .venv already exists. Using the existing one.
)

REM ----- Upgrade pip -----
echo [INFO] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip

REM ----- Install requirements -----
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt was not found in this folder.
    pause
    exit /b 1
)

echo [INFO] Installing packages from requirements.txt...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Installation failed. Please check the error messages above.
    pause
    exit /b 1
)

echo ============================================================
echo  Setup complete! You can now run start_server_ngrok.bat
echo ============================================================
pause