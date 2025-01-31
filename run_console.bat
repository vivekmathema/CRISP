@echo off
@ ECHO ########    #########     ####    ########    ######### ##########
@ ECHO ##    ##    ##     ##      ##     ##    ##    ##     ##   ## ##
@ ECHO ##          ##     ##      ##     ##          ##     ##   ## ##
@ ECHO ##          #######        ##     ########    #########   ## ##
@ ECHO ##          ##    ##       ##           ##    ##          ## ##
@ ECHO ##    ##    ##     ##      ##     ##    ##    ##          ## ##
@ ECHO ########    ##      ##    ####    ########    ##        ##########
@ ECHO ------------------------------------------------------------------------- 
@ ECHO Running CRISPII in Windows Console mode
@ ECHO For batch operation using configuration file. Default location: crisp_root/config/
@ ECHO For command line syntax use syntax:    run_console.bat  crisp.pyc --help
@ ECHO Example on running classifier inference with default settings and models is given below
@ ECHO Example syntax:  run_console.bat crisp.py --run_session  cls_inf   --config_fpath  ./config/default.conf 
@ ECHO ========================================================================== 


%~dp0.\python3cpu_env\scripts\python.bat %* crisp.py --help
pause