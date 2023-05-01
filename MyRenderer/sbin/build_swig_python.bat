@echo off
cd %~dp0
echo Starting Build Swig Python...
echo ===========================================================
python -u build_swig_python.py
if not %errorlevel% == 0 (
    echo The program abnormal exited with %errorlevel%.
    pause
)
echo ===========================================================
echo Build Swig Python Ended