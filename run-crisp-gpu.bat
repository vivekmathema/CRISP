@echo off
REM Running CRISPII GPU version from internal python environment Windows 10 version
REM Requires CUDA 11.0+ cuDNN 8.0+ library for running on NVIDIA RTX2000s GPUs
REM "Reset Setting" option to start default configuration
%~dp0.\python3gpu_env\scripts\python.bat crisp.py
pause