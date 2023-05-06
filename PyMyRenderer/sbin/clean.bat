@echo off
cd %~dp0
echo Import PYD...
echo ===========================================================
rm -r -f ../kamanri

if not %errorlevel% == 0 (
    echo The program abnormal exited with %errorlevel%.
    pause
)
echo ===========================================================
echo Import PYD Ended