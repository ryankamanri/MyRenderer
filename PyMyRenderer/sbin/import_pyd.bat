@echo off
cd %~dp0
echo Import PYD...
echo ===========================================================
cp ../../MyRenderer/build/windows-default/Kamanri.py ../kamanri
cp ../../MyRenderer/build/windows-default/Debug/_swig_kamanri.pyd ../kamanri

if not %errorlevel% == 0 (
    echo The program abnormal exited with %errorlevel%.
    pause
)
echo ===========================================================
echo Import PYD Ended