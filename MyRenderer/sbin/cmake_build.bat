@echo off
cd %~dp0
echo Starting CMake Build...
echo ===========================================================
cmake -S ../ -B ../build/windows-default
if not %errorlevel% == 0 (
    echo The program abnormal exited with %errorlevel%.
    pause
)
echo ===========================================================
echo CMake Build Ended
