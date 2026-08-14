@echo off
REM ==============================================================================
REM ASL SPECULATIVE ONNX INFERENCE ENGINE — BUILD SCRIPT (WINDOWS / GCC / MSVC)
REM
REM PREREQUISITES:
REM   1. Download ONNX Runtime C++ package from:
REM      https://github.com/microsoft/onnxruntime/releases
REM   2. Set ORT_DIR to the extracted onnxruntime directory, e.g.:
REM      set ORT_DIR=C:\onnxruntime-win-x64-1.21.0
REM   3. Optionally set ORT_DML=1 to link DirectML for Intel iGPU acceleration
REM ==============================================================================

echo Building ASL C++ Speculative ONNX Runtime Inference Engine...
echo.

REM ── Auto-detect ONNX Runtime path ───────────────────────────────────────────
IF "%ORT_DIR%"=="" (
    FOR /D %%D IN ("%USERPROFILE%\onnxruntime*" "C:\onnxruntime*" ".\onnxruntime*") DO (
        IF EXIST "%%D\include\onnxruntime_cxx_api.h" (
            SET "ORT_DIR=%%D"
            GOTO :FOUND_ORT
        )
    )
    echo [!] ONNX Runtime not found. Please set ORT_DIR:
    echo     set ORT_DIR=C:\path\to\onnxruntime-win-x64-VERSION
    echo     Download from: https://github.com/microsoft/onnxruntime/releases
    goto :ERROR
)
:FOUND_ORT
echo [+] ONNX Runtime: %ORT_DIR%
echo.

SET "INCLUDE_FLAGS=-I"%ORT_DIR%\include""
SET "LIB_FLAGS=-L"%ORT_DIR%\lib" -lonnxruntime"
SET "SRC=inference_engine.cpp"
SET "OUT=inference_engine.exe"

REM ── Try GCC (g++) ────────────────────────────────────────────────────────────
where g++ >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [+] Detected GCC (g++). Building with -O3 -mavx2 -mfma C++17...
    g++ -O3 -mavx2 -mfma -std=c++17 %INCLUDE_FLAGS% %SRC% -o %OUT% %LIB_FLAGS%
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo [+] Build SUCCESS: %OUT%
        echo [*] Usage:
        echo     %OUT% asl_encoder.onnx asl_decoder.onnx asl_draft.onnx 2484
        echo [*] Copy %ORT_DIR%\lib\onnxruntime.dll next to inference_engine.exe
    ) else (
        echo [!] GCC build failed. Check ORT_DIR and include paths.
        goto :ERROR
    )
    goto :END
)

REM ── Try MSVC (cl.exe) ────────────────────────────────────────────────────────
where cl >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [+] Detected MSVC (cl.exe). Building with /O2 /arch:AVX2 C++17...
    SET "ORT_LIB=%ORT_DIR%\lib\onnxruntime.lib"
    cl /O2 /arch:AVX2 /std:c++17 /EHsc /I"%ORT_DIR%\include" %SRC% /Fe:%OUT% "%ORT_LIB%"
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo [+] Build SUCCESS: %OUT%
        echo [*] Usage:
        echo     %OUT% asl_encoder.onnx asl_decoder.onnx asl_draft.onnx 2484
        echo [*] Copy %ORT_DIR%\lib\onnxruntime.dll next to inference_engine.exe
    ) else (
        echo [!] MSVC build failed. Ensure you're in a Developer Command Prompt.
        goto :ERROR
    )
    goto :END
)

echo [!] Neither g++ nor cl.exe found. Install MinGW-w64 or Visual Studio Build Tools.
:ERROR
exit /b 1

:END
exit /b 0
