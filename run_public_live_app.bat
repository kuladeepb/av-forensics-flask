@echo off
title AV-Forensics Live Server & Public Tunnel
echo =======================================================================
echo          AV-Forensics: 1-Click Live Public Deployment
echo =======================================================================
echo.

cd /d "%~dp0"

echo [1/3] Detecting Python environment...
set PYTHON_CMD=python
where python >nul 2>&1
if errorlevel 1 (
    where py >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=py -3
    ) else (
        if exist "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3824.0_x64__qbz5n2kfra8p0\python3.13.exe" (
            set PYTHON_CMD="C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3824.0_x64__qbz5n2kfra8p0\python3.13.exe"
        ) else (
            set PYTHON_CMD="C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe"
        )
    )
)

echo [2/3] Starting local Flask server on http://127.0.0.1:5000 ...
start "AV-Forensics Flask Backend" cmd /k "%PYTHON_CMD% app.py"

echo Waiting 3 seconds for backend to start...
timeout /t 3 /nobreak >nul

echo [3/3] Launching live public HTTPS tunnel (Cloudflare)...
echo =======================================================================
echo Look for your public URL below (ending in '.trycloudflare.com'):
echo Share that link with anyone on their phone or laptop!
echo =======================================================================
echo.
"%~dp0cloudflared.exe" tunnel --url http://127.0.0.1:5000

pause
