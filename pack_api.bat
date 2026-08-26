@echo off
echo ===================================================
echo Building Portable Sign Language API Executable Package
echo ===================================================

set REPOS_DIR=%~dp0
set API_PROJECT=%REPOS_DIR%SignLanugaApi\SignLanguageApi.csproj
set OUTPUT_DIR=%REPOS_DIR%SignLanguageApi_Executable_Package
set ZIP_OUTPUT=%REPOS_DIR%SignLanguageApi_Portable.zip

echo [1/4] Publishing Self-Contained API for Windows x64...
dotnet publish "%API_PROJECT%" -c Release -r win-x64 --self-contained true -o "%OUTPUT_DIR%" -p:PublishSingleFile=false -p:PublishAot=false

if %errorlevel% neq 0 (
    echo [ERROR] dotnet publish failed!
    exit /b %errorlevel%
)

echo [2/4] Copying required runtime assets...
if exist "%REPOS_DIR%SignLanugaApi\asl_sovereign_zenith_v22.onnx" (
    copy /y "%REPOS_DIR%SignLanugaApi\asl_sovereign_zenith_v22.onnx" "%OUTPUT_DIR%\"
)
if exist "%REPOS_DIR%SignLanugaApi\hand_landmarker.task" (
    copy /y "%REPOS_DIR%SignLanugaApi\hand_landmarker.task" "%OUTPUT_DIR%\"
)
if exist "%REPOS_DIR%SignLanugaApi\SignLanguageApp.db" (
    copy /y "%REPOS_DIR%SignLanugaApi\SignLanguageApp.db" "%OUTPUT_DIR%\"
)
if exist "%REPOS_DIR%SignLanugaApi\scripts" (
    xcopy /s /y /i "%REPOS_DIR%SignLanugaApi\scripts" "%OUTPUT_DIR%\scripts"
)
if exist "%REPOS_DIR%SignLanugaApi\MEDIA" (
    xcopy /s /y /i "%REPOS_DIR%SignLanugaApi\MEDIA" "%OUTPUT_DIR%\MEDIA"
)
if exist "%REPOS_DIR%SignLanugaApi\VIDEO" (
    xcopy /s /y /i "%REPOS_DIR%SignLanugaApi\VIDEO" "%OUTPUT_DIR%\VIDEO"
)
if exist "%REPOS_DIR%SignLanugaApi\wwwroot" (
    xcopy /s /y /i "%REPOS_DIR%SignLanugaApi\wwwroot" "%OUTPUT_DIR%\wwwroot"
)

echo [3/4] Creating launcher START_API.bat...
(
echo @echo off
echo echo Starting Sign Language API Portable...
echo cd /d "%%~dp0"
echo start "Sign Language API" SignLanguageApi.exe
echo echo API Started on http://localhost:5179 / https://localhost:7084
) > "%OUTPUT_DIR%\START_API.bat"

echo [4/4] Creating Zip Archive...
powershell -Command "Compress-Archive -Path '%OUTPUT_DIR%\*' -DestinationPath '%ZIP_OUTPUT%' -Force"

echo ===================================================
echo Packaging Completed Successfully!
echo Executable Package Path: %OUTPUT_DIR%
echo Portable Zip Archive: %ZIP_OUTPUT%
echo ===================================================
