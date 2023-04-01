@echo off
cd %~dp0
echo Starting Clean...
echo ===========================================================
rm -r -f ../build
if not %errorlevel% == 0 (
    echo The program abnormal exited with %errorlevel%.
    pause
)
echo ===========================================================
echo Clean Ended

