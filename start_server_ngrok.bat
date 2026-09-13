@echo off
REM ============================================================
REM  start_server_ngrok.bat
REM  Launches the AI Duplicate Detection Service (Prototype).
REM  Uses relative paths so it runs on any machine without
REM  editing.
REM ============================================================

REM %~dp0 = the folder this .bat file lives in
set "PROJECT_DIR=%~dp0"

REM Strip trailing backslash (if any) for cleaner echo output
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

echo Starting AI Duplicate Detection Service (Prototype)...
echo Project directory: %PROJECT_DIR%

REM ----- Check that the venv is ready -----
if not exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Please run install_requirements.bat first.
    pause
    exit /b 1
)

start "AI Duplicate Detection Backend" cmd /k "cd /d "%PROJECT_DIR%" && .venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo Service is launching...
echo Swagger UI Docs: http://localhost:8000/docs