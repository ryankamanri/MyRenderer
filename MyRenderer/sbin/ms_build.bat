@echo off
cd %~dp0
echo Starting MS Build...
echo ===========================================================
MSBuild ../build/windows-default/MyRenderer.sln
if not %errorlevel% == 0 (
    echo The program abnormal exited with %errorlevel%.
    pause
)
echo ===========================================================
echo MS Build Ended
