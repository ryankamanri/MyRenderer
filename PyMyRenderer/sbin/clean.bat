@echo off
cd %~dp0
echo Import PYD...
echo ===========================================================
rm ../kamanri/_swig_kamanri.pyd
rm ../kamanri/Kamanri.py

if not %errorlevel% == 0 (
    echo The program abnormal exited with %errorlevel%.
    pause
)
echo ===========================================================
echo Import PYD Ended