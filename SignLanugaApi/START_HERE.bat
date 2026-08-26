@echo off
echo ==========================================
echo Starting Sign Language Suite...
echo ==========================================

:: Check if dotnet is installed
where dotnet >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] .NET CLI is not installed or not in PATH.
    echo Please install the .NET 10.0 runtime.
    pause
    exit /b 1
)

:: Start the API in a separate window
echo [INFO] Starting API...
start "Sign Language API" /D "%~dp0API" dotnet SignLanguageApi.dll

:: Give the API 3 seconds to spin up and bind to its ports
timeout /t 3 /nobreak >nul

:: Start the App
echo [INFO] Starting App...
start "Sign Language App" /D "%~dp0App" SignLanguageApp.exe

echo [INFO] Startup completed.
timeout /t 2 /nobreak >nul
