@echo off
cd %~dp0
echo Starting MyRenderer...
echo ===========================================================
cd ../build/windows-default/Debug
MyRenderer.exe
if not %errorlevel% == 0 (
    echo The program abnormal exited with %errorlevel%.
    pause
)
echo ===========================================================
echo MyRenderer Ended