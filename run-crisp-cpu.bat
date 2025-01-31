REM Running CRISP-II from internal python environment Windows 10 version
REM Load "Reset Setting" using default.conf to start default configuration
%~dp0.\python3cpu_env\scripts\python.bat crisp.py  
REM %~dp0.\python3cpu_env\scripts\python.bat crisp.py     --gui_type 1  --config_run 1  --run_session None

pause