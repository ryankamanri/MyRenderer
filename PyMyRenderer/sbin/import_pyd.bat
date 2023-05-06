@echo off
cd %~dp0
echo Import PYD...
echo ===========================================================
xcopy ..\..\MyRenderer\build\kamanri ..\kamanri /E /Y /I

if not %errorlevel% == 0 (
    echo The program abnormal exited with %errorlevel%.
    pause
)
echo ===========================================================
echo Import PYD Ended