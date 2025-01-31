#!/usr/bin/python
# CRISPII __init__
# CRISP II: Leveraging Deep Learning for Multiclass Phenotyping in GC×GC-TOFMS of End-Stage Kidney Disease Patients
# Mathema et al 2025 : Siriraj Metabolomics & Phenomics Center (SiMPC), Mahidol University, Bangkok, Thailand
# Software code under review VERSION
# encoding: utf-8
# A prt of source code file for the CRISPII. (Under Review)

'''
     ░▒▓██████▓▒░░▒▓███████▓▒░░▒▒▓█▓▒░░▒▓███████▓▒░▒▓███████▓▒░  ░▒▓█▓▒░ ▒▓█▓▒░
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒  ▒▓█▓▒░▒▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒  ░▒▓█▓▒░ ▒▓█▓▒░
    ░▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒░▒▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒  ░▒▓█▓▒░ ▒▓█▓▒░ 
    ░▒▓█▓▒░      ░▒▓███████▓▒░░▒▒▓█▓▒░░▒▓██████▓▒░░▒▓███████▓▒░  ░▒▓█▓▒░ ▒▓█▓▒░
    ░▒▓█▓▒░      ░▒▓█▓▒░░▒▒▓█▓░▒▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░        ░▒▓█▓▒░ ▒▓█▓▒░ 
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▒▓█▓░▒▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░        ░▒▓█▓▒░ ▒▓█▓▒░
     ░▒▓██████▓▒░░▒▓█▓▒░░▒▒▓█▓░▒▒▓█▓▒░▒▓███████▓▒░░▒▓█▓▒░        ░▒▓█▓▒░ ▒▓█▓▒░
------------------------------------------------------------------------------------------------------------------------------------------------------------------
__author__      = "Mathema VB,Jariyasopit N, Phochmak T, Wanichthanarak K, Werayachankul T, Sathirapongsasuti N, Kitiyakara C, Sirivatanauksorn Y, Khoomrung S*   |
__institute__   = "Metabolomics and Systems Biology,Department of Biochemistry,Faculty of Medicine Siriraj Hospital, Mahidol University, Bangkok 10700, Thailand" |
__license__     = "MIT"                                                                                                                                           |
__version__     = "0.02"                                                                                                                                          |
__maintainer__  =  "VBM"                                                                                                                                          |
__email__       = "(under review)"                                                                                                                                |
__status__      = "Under review"                                                                                                                                  |
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Software code for the part of the manuscript (under review) entitled:
CRISP II: Leveraging Deep Learning for Multigroup Classification in GC×GC-TOFMS of End-Stage Kidney Disease Patients

==============================================================================================================
A Typical folder structure for running CRISPII

CRISP_ROOT  (main base folder)
|
|-----python3_env   (optional if using the dependencies pre-installed python3_env for Windows 10 / 11 OS ) 
|-----assets
|-----config
|-----datasets
|-----roids_data
|-----gan_data
|-----classifier_data
|-----output
|-----logs
|-----temp
|__init__.py
|crisp.py            
|autoroi.py
|UIutils.py
|utils_tools.py
|cmd_args.py
|crisp_ui.py
|VGG_module.py
|fid_module.py
|iafg_runner.py
|roi_selector.py
|siamese_net.py
|requirements_cpu.txt
|requirements_gpu.txt
|run_console.bat       
|run-crisp-gpu.bat  
|run-crisp-cpu.bat     
 
==============================================================================================================

#Part of the software associtaed with manuscript entitled: "CRISP: A Deep Learning Architecture for GC×GC-TOFMS Contour ROI Identification, Simulation, and Analysis in Imaging Metabolomics"

'''