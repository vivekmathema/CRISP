#!/usr/bin/python
# CRISP II: Leveraging Deep Learning for Multigroup Classification in GC×GC-TOFMS of End-Stage Kidney Disease Patients
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
#============================================

import numpy as np
import os, sys, time
from pathlib import Path
import shutil
import random
import configparser
import glob
import logging
import pickle
#================================================================================Pyqt5 Standard Libraries and dependent libs
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import  *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap
# from crisp_ui  import Ui_MainWindow                       # for importing CRISP UI ( full Ui will be released after accepatnce of manuscript)
                                     
#==========================================================================Fro GPU memory cleaning
from numba import cuda
#==========================================================================Import custom libraries
from cmd_args import *
from utils_tools import *
from colorama import *             
from tqdm import tqdm 
#==============
init(autoreset=True)


def to_bool(s):
    return 1 if s.lower() == 'true' else 0

#class BaseClass(QMainWindow):

class BaseClass(QtWidgets.QMainWindow ):   #  (QtWidgets.QMainWindow, Ui_MainWindow):            
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        if os.path.isfile("./crisp_ui.py"):
            from crisp_ui  import Ui_MainWindow
            Ui_MainWindow.__init__(self)     # load from .py if coutour_gui.py  exists
            self.logger("[%s] | GUI loaded from code"%time_stamp())
        else:
            loadUi("crisp_ui.ui",self)
            self.logger("[%s] | GUI loaded from UI template"%time_stamp())

        #===================================================== Conour theme defaulttheme
        self.setStyleSheet("") #("background-color: lightblue;")
        #self.setWindowFlags(Qt.FramelessWindowHint) 

        self.actionClear_Terminal_output.triggered.connect(self.clear_terminal_view)                                         # for clearing terminal view
        self.actionClear_Terminal_output.setShortcut('Ctrl+B')
        self.actionOpen_root_folder.triggered.connect(lambda dir_path: self.explore_dir("."))                               # Open current CRISP root fodler
        self.actionView_batch_extracted_singleROI_folder.triggered.connect( lambda dir_path: self.explore_dir(self.org_dim_rois_extract.toPlainText()))
        self.actionOpen_GAN_synthesis_image_folder.triggered.connect( lambda dir_path: self.explore_dir(self.gan_output_path.toPlainText()))
        self.action_inference_tagged_image_output_folder.triggered.connect( lambda dir_path: self.explore_dir(self.classifier_output_path.toPlainText()))
        self.actionOpen_inference_report_folder.triggered.connect( lambda dir_path: self.explore_dir(self.classifier_report_path.toPlainText()))

        #===================================Trigerred panel 
        self.actionExtract_ROIs.triggered.connect(self.orgdim_rois_extract)
        self.actionProcess_multiclass_contours.triggered.connect(self.preprocess_multiclass_images)

        self.actionProcess_IAFG.triggered.connect(self.run_IAFG)
        self.actionProcess_multiclass_AFRC.triggered.connect(self.run_IAFG_multiclass)

        self.actionProcess_ROIs_for_single_pair_AFRC.triggered.connect(self.auto_compute_ssim_roi)
        self.actionProcess_ROIs_for_multiclass_AFRC.triggered.connect(lambda show_msg: self.multiclass_auto_compute_ssim_roi(True))

        self.actionConstruct_DeepStacking_db.triggered.connect(self.make_stacked_dataset)
        self.actionBuild_multiclass_deepstacking_dataset.triggered.connect(self.multiclass_make_stacked_dataset)


        #======================== Traiend model list fro GAN
        self.roids_model_list.setGeometry(QtCore.QRect(120,90,211, 0))                     # make invisible  self.roids_model_list.setGeometry(QtCore.QRect(90, 90, 191, 120))
        self.roids_model_pushbutton.clicked.connect(self.show_roids_model_list)
        self.roids_model_list.doubleClicked.connect(self.set_roids_model_id) 
        self.roi_ds_id.textChanged.connect(self.update_roids_fname)                       # roi_id_text_chnage

        #======================== Traiend model list fro GAN
        self.gan_model_list.setGeometry(QtCore.QRect(410, 245, 151, 0))                     # make invisible
        self.gan_model_pushbutton.clicked.connect(self.show_gan_model_list)
        self.gan_model_list.doubleClicked.connect(self.set_gan_model_id) 

        #======================== Traiend model list for CLS
        self.cls_model_list.setGeometry(QtCore.QRect(290, 30, 261, 0))                     # make invisible  400, 310, 191, 0
        self.cls_model_pushbutton.clicked.connect(self.show_cls_model_list)
        self.cls_model_list.doubleClicked.connect(self.set_cls_model_id)
        #======================== CLassifier dataset cosntruction module
        self.construct_cls_dataset.clicked.connect(self.make_cls_dataset)
        self.set_cls_src_path.clicked.connect(self.set_classifier_src_path)
        self.set_cls_dst_path.clicked.connect(self.set_classifier_dst_path)
        self.set_cls_entire_path.clicked.connect(self.set_classifier_entire_src_path)  

        #=================================
        self.qm        = QtGui.QMessageBox
        #=================================
        self.run_mode  = None
        self.update_vars()
        try:                                                                                 # try not to generate fatal error if could not run logo images fro massets
            self.prv_logo ="./assets/logo.jpg"                                              # for preview wfwfct only
            self.loadImage(self.prv_logo)                                                    # load logo image self.image is loaded here in this module
            self.roi_preview_frame.setPixmap(QPixmap.fromImage(self.displayImage(cv2.imread(self.prv_logo))))  # dislay logo 
        except:
            self.logger("[%s] | Warning! Failed to load LOGO image from:  ./assets/logo.jpg "%time_stamp())
            pass

        self.roi_imagepath       = ""
        self.use_computed_roi    = False
        #=================================
        self.trainLoadImgBtn.clicked.connect(self.start_train)
        self.btn_make_contour.clicked.connect(self.gen_contours)
        self.ExitBtn.clicked.connect(self.Exit_CRISP)
        self.ExitBtn.setShortcut('Ctrl+Q')
        #=================================                                                # save copnfig buttons
        self.roids_saveconfig_btn.clicked.connect(self.store_gan_config)
        self.gan_saveconfig_btn.clicked.connect(self.store_gan_config)
        self.cls_saveconfig_btn.clicked.connect(self.store_gan_config)
        self.multiclass_saveconfig_btn.clicked.connect(self.store_gan_config)
        #=================================# Menu trigger buttons for GUI menus
        self.actionSave_entire_settings.triggered.connect(self.config_store)
        self.actionSave_entire_settings.setShortcut('Ctrl+S')
        self.actionLoad_settings.triggered.connect(self.select_custom_configfile)
        self.actionLoad_previous_setting.triggered.connect(self.load_model_config) 
        self.actionLoad_default_settings.triggered.connect(self.load_default_config)       
        self.actionSave_settings_as.triggered.connect(self.save_custom_config)            # save settings as 
        self.actionProcess_IAFG.triggered.connect(self.run_IAFG)           
        self.actionExtract_ROIs.triggered.connect(self.orgdim_rois_extract)          
        self.actionConstruct_DeepStacking_db.triggered.connect(self.make_stacked_dataset)
        self.restore_ssim_data_btn.clicked.connect(self.load_ssim_data)

        #=======================================                                           # Theme settings
        self.actionDark_Theme.triggered.connect(self.set_dark_theme)
        self.actionDaylight_Theme.triggered.connect(self.set_light_theme)
        self.actionDefault_Theme.triggered.connect(self.set_default_theme)
        #=======================================

        self.actionTrain_GAN_generator.triggered.connect(self.start_train)                # GAN train syntehsis
        self.actionRun_GAN_generator.triggered.connect(self.gen_contours)

        self.actionTrain_Classifier.triggered.connect(self.train_class)                   # Classifier train model              
        self.actionRun_inferencing.triggered.connect(self.run_class)
        self.actionAbout_ContourGANs.triggered.connect(self.about_crisp)
        self.actionOnline_Help_Github.triggered.connect(self.crisp_online)
        self.actionExitProgram.triggered.connect(self.Exit_CRISP)                       # Exits the program without saving...
        self.actionExitProgram.setShortcut('Ctrl+Q')

        #=================================
        self.Set_model_path.clicked.connect(self.set_model_folder)
        self.Set_result_path.clicked.connect(self.set_result_folder)
        self.Set_preview_folder.clicked.connect(self.set_preview_folder)
        self.Select_data_folder.clicked.connect(self.set_data_image)
        #=================================
        self.roids_setpath.clicked.connect(self.set_roids_path)
        self.gan_roi_template_btn.clicked.connect(self.set_roi_preview)
        self.refresh_manual_roi_btn.clicked.connect(self.get_roi_from_coordinates)
        self.gan_set_roi_btn.clicked.connect(self.get_roi_from_template)
        self.gan_view_cropped_template_btn.clicked.connect(self.show_cropped_roi_image)
        self.gan_preview_org_roi_template_btn.clicked.connect(self.show_org_roi_image)
        self.gan_set_processed_image_path_btn.clicked.connect(self.set_processed_image_path)
        self.apply_auto_roi_btn.clicked.connect(self.apply_auto_roi)
        self.auto_roi_compute_btn.clicked.connect(self.auto_compute_ssim_roi)
        #==== multiclass
        self.btn_preprocess_multiclass.clicked.connect(self.preprocess_multiclass_images)                    # to pre-process multiclass extraction
        self.btn_muticlass_source_dataset.clicked.connect(self.set_muticlass_source_dataset) 
        self.multiclass_auto_roi_compute_btn.clicked.connect(self.multiclass_auto_compute_ssim_roi) 
        #============   STAY on TOP  toggle
        self.actionStay_on_Top.triggered.connect(self.set_ui_stayontop)

        self.btn_preprocess_singleclass_images.clicked.connect(self.orgdim_rois_extract)                     # for single class extraction
        self.set_ds_restore_file.clicked.connect(self.set_ds_restore_filepath)
        #============IAFG
        self.process_iafg_btn.clicked.connect(self.run_IAFG)
        self.Set_contourA_btn.clicked.connect(self.set_ContourA_img)
        self.Set_contourB_btn.clicked.connect(self.set_ContourB_img)
        self.iafg_rnd_select_btn.clicked.connect(self.set_random_afrc_images)
        self.btn_set_multiclass_val_cohort_path.clicked.connect(self.set_multiclass_val_cohort_path)


        #============ for multiclass afrc
        self.set_ds_src_dataset.clicked.connect(self.set_pth_multiclass_src_dataset)
        self.set_ds_control_id.clicked.connect(self.set_pth_multiclass_control_group)  
        self.btn_preview_multiclass_struct.clicked.connect(self.preview_multiclass_folder_list) 
        self.multiclass_afrc_groups_list.itemClicked.connect(self.multiclass_afrc_groups_list_item_clicked)
        self.btn_process_multiclass_afrc.clicked.connect(self.run_IAFG_multiclass)

        #============ for multiclass deepstacking
        self.set_multiclass_ds_dataset_path.clicked.connect(self.set_multiclass_ds_dataset)
        self.set_multiclass_ds_control.clicked.connect(self.set_multiclass_ds_control_group)  
        self.set_multiclass_ds_pos_file.clicked.connect(self.set_multiclass_ds_pos_fname) 
        self.multiclass_deepstack_groups_list.itemClicked.connect(self.multiclass_deepstacking_groups_list_item_clicked)
        self.btn_multiclass_deepstacking_preview.clicked.connect(self.preview_multiclass_deepstacking_folder_list)
        #==============================
        #===================================ROIs related SSIM item list box
        self.Set_contourA_ds_btn.clicked.connect(self.set_contourA_dsimg_path)
        self.Set_contourB_ds_btn.clicked.connect(self.set_contourB_dsimg_path) 
        self.SSIM_list.doubleClicked.connect(self.set_SSIM_value)                                # for single class ROIs value 
        self.multiclass_SSIM_list.doubleClicked.connect(self.set_multiclass_SSIM_value)          # For multiclass ROIS Value
        self.show_computed_roi.stateChanged.connect(self.set_updated_gui_opts)
        self.multiclass_show_computed_roi.stateChanged.connect(self.set_updated_gui_opts)
        self.apply_deep_stack_btn.clicked.connect(self.make_stacked_dataset)
        self.set_roi_ext_path_btn.clicked.connect(self.set_orgdim_rois_path)
        self.roi_dims = None
        self.multiclass_roi_dims = None                                                                                       # holds the dimension of teh ROIs, need to be initalized here so thatit does not get reset by program eachtime
        self.btn_multiclass_deepstack_dataset_builder.clicked.connect(self.multiclass_make_stacked_dataset)                   # multiclass_dataset builder 
        #===================================ROIs related SSIM item list box
        self.SSIM_list.itemSelectionChanged.connect(self.set_SSIM_value)
        #=================================Classifier training
        self.classifier_train.clicked.connect(self.train_class)
        self.classifier_run.clicked.connect(self.run_class)
        self.classifier_src_images_btn.clicked.connect(self.set_classifier_source_data)
        self.classifier_model_path_btn.clicked.connect(self.set_classifier_model_path)
        self.classifier_testing_images.clicked.connect(self.set_classifier_test_image_path)
        self.cls_classification_report_btn.clicked.connect(self.set_classifier_report_path)
        self.classifier_result_output_btn.clicked.connect(self.set_classifier_output_path)
        self.btn_set_pred_heatmap_store.clicked.connect(self.set_pred_heatmap_store_path)                                       # to DO for real time changhe
        #================================ Temproary
        if args.config_run == False:
            self.show_message_box()                                                         # for restore message  display only if run in GUI mode not CLI mode


    def set_light_theme(self):                                                              # THEMES OF SKIN
        self.setStyleSheet("background-color: lightblue;")                                     # change background theme colour

    def set_dark_theme(self):    
        self.setStyleSheet("background-color: darkgrey;")                                   # change background theme colour

    def set_default_theme(self):
    	self.setStyleSheet('')                                                              # default colour

    #==================================================================                     # ROIDS _list info

    def set_roids_model_id(self):
        self.roi_ds_id.setText(self.roids_model_list.currentItem().text() )
        self.roids_model_list.setGeometry(QtCore.QRect(120, 90, 211, 100))                    # hide list after selection
        self.logger("[%s] | Selected ROI-DS model ID: %s"%(time_stamp(),self.roids_model_list.currentItem().text()) )
        self.ds_pos_filepath.setText(self.roids_src_path.toPlainText() + "/"+ self.roids_model_list.currentItem().text() +".data")

    def show_roids_model_list(self):

        if not self.roids_model_pushbutton.isChecked():
            self.roids_model_list.setGeometry(QtCore.QRect(120, 90, 211, 0))
            return

        self.roids_model_list.setGeometry(QtCore.QRect(120, 90, 211, 100))                  # show list

        model_name =glob.glob(self.roids_src_path.toPlainText() +"/*.data")               # get list of models
        self.roids_model_list.clear()
        for each_model in model_name:
            each_model = os.path.splitext(os.path.basename(each_model))[0]             # get basefilename and strip and use without extension
            self.roids_model_list.addItem(each_model)     


    #==================================================================

    def set_gan_model_id(self):
    	self.model_id_box.setText(self.gan_model_list.currentItem().text() )
    	self.gan_model_list.setGeometry(QtCore.QRect(410, 245, 151, 0))                    # hide list after selection
    	self.logger("[%s] | Selected GAN model: %s"%(time_stamp(),self.gan_model_list.currentItem().text()) )


    def show_gan_model_list(self):

        if not self.gan_model_pushbutton.isChecked():
            self.gan_model_list.setGeometry(QtCore.QRect(410, 245, 151, 0))
            return

        self.gan_model_list.setGeometry(QtCore.QRect(410, 245, 151, 120))                  # show list

        model_name =glob.glob(self.gan_model_path.toPlainText() +"/*.model")               # get list of models
        self.gan_model_list.clear()
        for each_model in model_name:
            each_model = os.path.splitext(os.path.basename(each_model))[0]                 # get basefilename and strip and use without extension
            self.gan_model_list.addItem(each_model)    	


   	#============================================================

    def set_cls_model_id(self):
    	self.classifier_model_id.setText(self.cls_model_list.currentItem().text() )
    	self.cls_model_list.setGeometry(QtCore.QRect(290, 30, 261, 0))                     # hide list after selection
        # Show class within trained models
    	try:
            print("[%s] | Classifier model :%s"%(time_stamp(), self.classifier_model_path.toPlainText()  +"/"+ self.classifier_model_id.toPlainText() +".model"))
            with open(self.classifier_model_path.toPlainText()  +"/"+ self.classifier_model_id.toPlainText() +".model" , 'r') as model_id:   # read model ID file for classification labels
                class_list = model_id.read().strip().split(",")
                print("\nSelected Model Id: %s"%self.classifier_model_id.toPlainText(), "\nTraiend classes  :",class_list)
                #print()
                self.gan_current_loss_display.insertPlainText("\nClass Labels    : "+ " , ".join(class_list))
                self.gan_current_loss_display.insertPlainText("\nClasses Trained: [%d]\nModel ID          : %s"%(len(class_list),self.classifier_model_id.toPlainText()))
                
    	except:
            self.logger('[%s] Warning! Unable to read the model Labels. Please refer to the classification code in corrsponding tarining dataset'%time_stamp(), color = 5 )


    def show_cls_model_list(self):
        if not self.cls_model_pushbutton.isChecked():
            self.cls_model_list.setGeometry(QtCore.QRect(290, 30, 261, 0))
            return


        self.cls_model_list.setGeometry(QtCore.QRect(290 , 30, 261, 120))                   # show the list item
    	
        model_name =glob.glob(self.classifier_model_path.toPlainText() +"/*.model")
        self.cls_model_list.clear()
        for each_model in model_name:
            each_model = os.path.splitext(os.path.basename(each_model))[0]                 # get basefilename and strip and use without extension
            self.cls_model_list.addItem(each_model)

    #==============================================================


    def about_crisp(self):
    	msg  ="\nCRISP: GCxGC-TOFMS Contour ROI Identification, Simulation & untargeted metabolomics Profiler (under Review)" 
    	msg  +="\nDeclaimer: There may still be few bugs in the GUI code, which we will be constantly updating to fix such issues" 
    	msg +="\n\nMathema et al 2021, Siriraj Metabolomics & Phenomics Center (SiMPC), Mahidol University, Bangkok, Thailand"
    	self.qm.about(self, "CRISP v0.1", msg)

    def crisp_online(self):
    	try:
    		import webbrowser
    		webbrowser.open("https://github.com/vivekmathema/CRISP-II")
    		self.logger("[%s] | Opening weblink: https://github.com/vivekmathema/CRISP-II"%time_stamp() , color = 2)
    	except:
    		self.logger("[%s] | Warning! Unable to open the weblink\n try manual browsing of : [ https://github.com/vivekmathema/CRISP-II ] "%time_stamp(), color =5 )
    		pass


    def init_logger(self, fname = "./log_file.txt", level=logging.INFO, shutdown = False):
        if shutdown:
            print("[%s] | Logging terminated..."%time_stamp())
            logging.shutdown()
            return

        try:
            from importlib import reload
            logging.shutdown()
            reload(logging)
        except:
            pass    

        if self.actionActivate_log.isChecked() :
            os.makedirs("./logs") if not os.path.exists("./logs")  else None                                                                                  # makes log folder if not present
            fname = "./logs/loginfo_" + time_stamp().replace(" ", "_").replace(":", "-").replace(":","-").replace("\\","-").replace("/","-") + ".log"         # writet he log file name
            print("[%s] | Logging activated!   "%time_stamp())
            print("[%s] | Log file path                 : %s"%(time_stamp(), fname))
            logging.basicConfig(filename=fname,level=level)                          # write log to file if checked
            logging.info("\n----------------------[Log Data]--------------------------\n")
            logging.info("[%s] | Start logging...       "%time_stamp())   
            logging.info("[%s] | Logging file location  :  %s"%(time_stamp(), fname)) 
        else:
            print("[%s] | Logging is NOT activated!  "%time_stamp())
            print("[%s] | Output INFO to StdOut ONLY..."%time_stamp())
            logging.basicConfig(level=level)                                                 # do not write to file if Logging si not checked



    def logger(self, msg = None, show_info= True, level=logging.INFO , color = 0):
    	try:
    		print (Style.BRIGHT + Fore.WHITE + msg) if show_info and (self.actionActivate_log.isChecked() and color == 0)  else None          # print message in console  & also log
    		print (Style.BRIGHT + Fore.BLACK + msg) if show_info and (self.actionActivate_log.isChecked() and color == 1)  else None          # print message in console  & also log
    		print (Style.BRIGHT + Fore.BLUE + msg) if show_info and (self.actionActivate_log.isChecked() and color  == 2)  else None          # print message in console  & also log
    		print (Style.BRIGHT + Fore.GREEN + msg) if show_info and (self.actionActivate_log.isChecked() and color == 3)  else None          # print message in console  & also log
    		print (Style.BRIGHT + Fore.YELLOW + msg) if show_info and (self.actionActivate_log.isChecked() and color == 4)  else None          # print message in console  & also log
    		print (Style.BRIGHT + Fore.RED + msg) if show_info and (self.actionActivate_log.isChecked() and color    == 5)  else None          # print message in console  & also log
    	except:
    		print(msg) if show_info and self.actionActivate_log.isChecked()  else None                                                                        # print message in console  & also log

    	try:

            if level == logging.DEBUG:
                logging.debug(msg)

            elif level == logging.info(msg):
                logging.info(msg)

            elif level == logging.WARNING:
                logging.warning(msg)

            elif level == logging.ERROR:
                logging.error(msg)

            elif level == logging.CRITICAL:
                logging.critical(msg)
    	except:
            pass                                                                       # just in case                                        



    def set_updated_gui_opts(self):                                                   # updated values of options on GUI (e.g preview) without updating or resetting all vals
    	self.preview_auto_roi = bool(self.show_computed_roi.isChecked())
    	#== IAFG paramters
    	self.cimageA_path     = self.contourA_imagepath.toPlainText().strip()
    	self.cimageB_path     = self.contourB_imagepath.toPlainText().strip()
    	self.iafg_fps         = int(self.iafg_fps_slider.value())
    	self.multiclass_iafg_fps  = int(self.multiclass_iafg_fps_slider.value())
    	self.iafg_duration    = int(self.iafg_duration_slider.value())
    	self.multiclass_iafg_duration    = int(self.multiclass_iafg_duration_slider.value())
    	self.auto_duration               = bool(self.multiclass_iafg_duration_flag_val.isChecked())
    	self.multiclass_auto_duration    = bool(self.iafg_duration_flag_val.isChecked())
    	self.auto_fps                    = bool(self.iafg_fps_flag_val.isChecked())
    	self.multiclass_auto_fps         = bool(self.multiclass_iafg_fps_flag_val.isChecked())
    	self.iafg_sharpen                = bool(self.iafg_sharpen_flag_val.isChecked())
    	self.multiclass_iafg_sharpen     = bool(self.multiclass_iafg_sharpen_flag_val.isChecked())
    	self.iafg_preview                = bool(self.iafg_preview_flag_val.isChecked())
    	self.multiclass_iafg_preview     = bool(self.multiclass_iafg_preview_flag_val.isChecked())
    	self.iafg_weight      = round(self.iafg_weight_val.value(), 3)
    	self.iafg_rnd_select  = bool(self.iafg_rnd_select_flag.isChecked())
    	self.orgdim_img_type  = self.orgdim_extraction_mode.currentText().strip()
    	self.orgdim_resize    = bool(self.resize_orgdim_flag.isChecked())
    	self.odim_width       = int(self.orgdim_width.value())
    	self.odim_height      = int(self.orgdim_height.value())
    	self.orgdim_roi_path  = self.org_dim_rois_extract.toPlainText().strip()
    	self.raw_file_path    = self.roi_template_image_path.toPlainText()
    	self.roi_imagepath    = self.roi_template_image_path.toPlainText()
    	self.raw_img_srcpath  = os.path.dirname(self.roi_template_image_path.toPlainText())
    	self.iagf_new_width   = int(self.iagf_width.value())
    	self.iagf_new_height  = int(self.iagf_height.value())  
    	self.grad_threshold   = int(self.grad_threshold_value.value())
    	self.autoset_cloned_multiclass = bool(self.flag_autoset_multiclass_dataset_path.isChecked())
        #=========================
    	self.multiclass_val_cohort_exists       = bool(self.multiclass_val_cohort_flag.isChecked())
    	self.multiclass_val_cohort_source       = self.multiclass_val_cohort.toPlainText()
    	self.multiclass_valid_cohort_pathname   = self.multiclass_val_cohort_path.toPlainText()
    	self.use_validation_cohot_scores        = bool(self.use_validation_cohot_scores_flag.isChecked())
        
        #===============Adaptive Thresholding
    	self.adaptive_trendline     = bool(self.show_adaptive_trendline_flag.isChecked())
    	self.adaptive_color         = self.adaptive_colormap.currentText()
    	self.adaptive_sort_mode     = self.adaptive_sort.currentText()
    	self.adaptive_threshold     = int(self.adaptive_threshold_value.value())
    	self.adaptive_kernel        = int(self.adaptive_kernel_value.value())  
    	self.adaptive_seed          = int(self.adaptive_suffle_seed.value())
        #====================================
    	self.multiclass_source_datapath = self.multiclass_source_dataset.toPlainText()
    	self.clone_multiclass_dataset   = bool(self.flag_clone_multiclass_dataset.isChecked())
    	self.pause_each_preporcessing   = bool(self.flag_pause_each_preprocessing.isChecked())

        #==========multiclass iafc
    	self.multiclass_iagf_new_width   = int(self.multiclass_iagf_width.value())
    	self.multiclass_iagf_new_height  = int(self.multiclass_iagf_height.value())
    	self.multiclass_pause_each_afrc  = bool(self.flag_multiclass_pause_classes.isChecked())        
    	self.pause_each_afrc             = bool(self.flag_pause_each_afrc.isChecked())
        #====================
    	self.iagf_resize_flag = bool(self.resize_iagf_output.isChecked())
    	self.multiclass_iagf_resize_flag = bool(self.multiclass_resize_iagf_output.isChecked())
    	self.orgdim_format    = self.orgdim_output_type.currentText()
    	self.roi_locked       = bool(self.lock_roi_val_flag.isChecked())
    	#=====================
    	self.iafr_dilation    =  bool(self.iafr_dilation_flag.isChecked())
    	self.multiclass_iafr_dilation    =  bool(self.multiclass_iafr_dilation_flag.isChecked())
    	self.iafr_erosion     =  bool(self.iafr_erosion_flag.isChecked())
    	self.multiclass_iafr_erosion     =  bool(self.multiclass_iafr_erosion_flag.isChecked())
    	self.iafr_noise       =  bool(self.iafr_noise_flag.isChecked())
    	self.multiclass_iafr_noise       =  bool(self.multiclass_iafr_noise_flag.isChecked())
    	self.iafr_denoise     =  bool(self.iafr_denoise_flag.isChecked())
    	self.multiclass_iafr_denoise     =  bool(self.multiclass_iafr_denoise_flag.isChecked())
    	#=====================
    	self.contourA_IAFGpth   = self.contourA_dspath.toPlainText()                    # path for Contour A related IAFG image 
    	self.contourB_IAFGpth   = self.contourB_dspath.toPlainText()
    	self.ssim_mode          = self.SSIM_mode_val.currentText().strip()
    	self.winsize            = self.scan_winsize_slider.value()
    	self.win_shift          = self.window_shift_slider.value()
    	self.ssim_thresthold    = round(self.ssim_thresthold_slider.value()/100 ,2)
    	self.win_segment        = self.scan_segment_slider.value()
    	self.stacking_flag      = self.use_stacking_flag.isChecked()
    	self.sharpen_autoroi    = self.sharpen_autoroi_flag.isChecked()
    	self.denoise_autoroi    = self.denoise_autoroi_flag.isChecked()
    	self.avoid_overlap      = self.avoid_stack_overlap_flag.isChecked()
    	#=======================multiclass
    	self.multiclass_ssim_mode          = self.multiclass_SSIM_mode_val.currentText().strip()
    	self.multiclass_winsize            = self.multiclass_scan_winsize_slider.value()
    	self.multiclass_win_shift          = self.multiclass_window_shift_slider.value()
    	self.multiclass_ssim_thresthold    = round(self.multiclass_ssim_thresthold_slider.value()/100 ,2)
    	self.multiclass_win_segment        = self.multiclass_scan_segment_slider.value()
    	self.multiclass_stacking_flag      = self.multiclass_use_stacking_flag.isChecked()
    	self.multiclass_sharpen_autoroi    = self.multiclass_sharpen_autoroi_flag.isChecked()
    	self.multiclass_denoise_autoroi    = self.multiclass_denoise_autoroi_flag.isChecked()
    	self.multiclass_avoid_overlap      = self.multiclass_avoid_stack_overlap_flag.isChecked()
    	self.multiclass_show_scan          = bool(self.multiclass_show_scan_roi_flag.isChecked())
    	self.multiclass_auto_roi_para_flag = bool(self.multiclass_auto_set_roi_parameter_flag.isChecked())
    	self.multiclass_preview_auto_roi   = bool(self.multiclass_show_computed_roi_flag.isChecked())
    	self.multiclass_store_plot         = bool(self.flag_multiclass_roi_graph_store.isChecked())
    	self.store_plot                    = bool(self.flag_roi_graph_store.isChecked())
    	self.clone_deepstack_dataset       = bool(self.flag_multiclass_deepstack_dataset_clone.isChecked())
        #=======================
    	self.sr_loss_val =0
    	self.sr_loss_list =[]
    	self.sr_blur_history = []
    	self.sr_preview_vertical = False
    	#========================
    	self.inference_thrshold      = self.cls_thrhld_slider.value()    
    	self.write_report_file       = bool(self.cls_write_output_flag.isChecked())
    	self.report_filetype         = self.cls_output_type.currentText()
    	self.report_filepath         = self.classifier_report_path.toPlainText()
    	self.class_verbosity         = int(self.cls_verbosity_level.currentText())
    	self.put_watermark           = bool(self.watermark_inference_flag.isChecked())
    	self.resize_watermarked      = bool(self.resize_watermark_flag.isChecked())
    	self.watermark_img_height    = int(self.watermarked_height_value.value())
    	self.watermark_img_width     = int(self.watermarked_width_value.value())
    	#=======================
    	self.cls_dataset_src_path    = self.cls_src_dataset_path.toPlainText()
    	self.cls_dataset_dst_path    = self.cls_dst_dataset_path.toPlainText()
    	self.cls_dataset_id          = self.cls_dataset_name.toPlainText()
    	self.cls_dataset_classID     = self.cls_dataset_class.toPlainText()
    	self.cls_train_ratio         = self.cls_train_ratio_slider.value()/100
    	self.cls_base_datasrc_path   = self.cls_entire_path.toPlainText()
    	self.build_entire_dataset    = bool(self.cls_entire_class_flag.isChecked())
    	self.clear_existing_files    = bool(self.clear_dst_files_flag.isChecked())
    	#======================= 
    	self.cls_agu_hflip_flag_value =      bool(self.cls_agu_hflip_flag.isChecked())  if self.cls_agu_hflip_flag.isChecked() else False
    	self.cls_agu_vflip_flag_value =      bool(self.cls_agu_vflip_flag.isChecked())  if self.cls_agu_vflip_flag.isChecked() else False
    	self.cls_img_shear_slider_value =    float( self.cls_img_shear_slider.value())  if bool(self.cls_agu_img_shear_flag.isChecked())  else 0.2 # default
    	self.cls_img_zoom_slider_value =     bool(self.cls_img_zoom_slider.value())     if bool(self.cls_img_zoom_flag.isChecked()) else 0.15 # default
    	self.cls_img_rot_slider_value  =     bool(self.cls_img_rot_slider.value() )     if bool(self.cls_img_rot_flag.isChecked()) else 5.0 # default
    	self.cls_set_img_fill_mode     =     self.cls_img_fill_mode_val.currentText()   if bool(self.cls_set_agu_fill_mode.isChecked()) else "nearest"  # default  fill mode
        #======================Prediction heatmap
    	self.show_pred_heatmap         =     bool(self.btn_cls_show_pred_heatmap_flag.isChecked())
    	self.cls_heatmap_level         =     self.cls_prediction_level.currentText().strip()
    	self.cls_heatmap_index         =     self.cls_prediction_level.currentIndex()
    	self.cls_heatmap_fpath         =     self.cls_store_pred_heatmap_path.toPlainText()
    	self.search_subfolders         =     bool(self.flag_inference_include_subfolders.isChecked())
    	self.show_inference_stats      =     bool(self.flag_show_basic_inference_stats.isChecked())
    	self.pause_each_heatmap        =     bool(self.btn_pause_each_heatmap_flag.isChecked())
    	#=======================
    	self.write_logfile             = bool(self.actionActivate_log.isChecked())



    def update_gan_parameters(self):
        #==================================      
        self.mode = self.output_shape_value  # default 512
        #==================================
        if self.mode == 512:
            self.scale_fac = 1
        elif self.mode == 256:
            self.scale_fac = 2      
        elif self.mode == 128:
            self.scale_fac = 4
        elif self.mode == 64:
            self.scale_fac = 8
        elif self.mode == 1024:                                                         # experimental, highly slow and  VRAM exhaustive ( try with BS : 4 on 11 GB+ VRAM GPU)
            self.scale_fac = 0.5    

        self.img_dim          = int(512/self.scale_fac)                                 # <-- 256 (change here) . multiply by scale_face
        if self.run_mode == "train_gan":
        	self.z_dim            = self.z_train_gan_dim
        else:
        	self.z_dim            = self.z_synth_gan_dim

        self.num_layers       = int(np.log2(self.img_dim)) - 3
        self.max_num_channels = self.img_dim * int(4 * self.scale_fac)      			# <--- img_dim * 8 (changehere )
        self.f_size           = self.img_dim // 2**(self.num_layers + 1)
        self.batch_size       = self.batch_size   # 16                                  # updates the core gane parameters that will be used everywhere                    # updates UI parameters only not all when needed
        self.blur_smp_flag    = bool(self.blur_sampling_flag.isChecked)
        self.blur_scr_lst     = []                                                      # hodl the  valeu for the blur scror s 
        self.avg_blr_history  = []                                                      # hodls teh vale for the average blur hsitory 
        self.write_report_file= self.cls_write_output_flag.isChecked()                  # teh write flag for the result report


    def update_vars(self):
        self.stored_model_pth   = self.gan_model_path.toPlainText()
        self.traning_iter       = int(self.epochNum_slider.value())
        self.autosave_freq      = self.autosave_epoch.value()
        self.gan_train_type     = self.gan_train_mode.currentText()         
        self.batch_size         = int(self.set_batch.currentText())
        self.source_image_path  = self.gan_src_image.toPlainText()
        self.preview_imgs       = self.preview_imgs_path.toPlainText()
        self.gen_imgs_path      = self.gan_output_path.toPlainText()
        self.sample_grid_dim    = int(self.gan_sample_dim.currentText())
        self.processor_type     = self.SelectProcessor.currentText().strip()
        self.img_preview        = int(self.preview_num.currentText())
        self.output_shape_value = int(self.synth_model_dim_selection.currentText())  if self.run_mode == "synthesis"\
                                                                             else int(self.gan_unit_shape.currentText())    # for making the dimension of synths                   
        self.prev_img_freq      = int(self.preview_img_freq.currentText())
        self.distortion_fac     = int(self.gan_distortion_fac.value())
        self.noise_intenisty    = int(self.gan_noise_intenisty.value())
        self.gan_int_synth_fac  = int(self.gan_intensity_fac.value())
        self.randomize_grad     = bool(self.grad_randomize_flag.isChecked())
        self.denois_val         = self.denoise_kernel_slider.value()
        self.sharpen_val        = self.sharpen_kernel_slider.value()
        self.blurred_val        = self.blur_kernel_slider.value()
        self.grad_kernel_size   = int(self.gradient_mode_slider_value.value())
        self.img_erosion_kernel = int(self.gan_erosion_kernel_slider.value())
        self.img_erosion_iter   = int(self.gan_erosion_iter_slider.value())
        self.img_dilate_kernel  = int(self.gan_dilation_kernel_slider.value())
        self.img_dilate_iter    = int(self.gan_dilation_iter_slider.value())
        self.processed_img_path = self.gan_processed_img_path.toPlainText().strip()
        self.gan_img_grid_dim   = int(self.gan_image_grid_val.currentText())
        self.synth_img_number   = int(self.synth_img_no_slider.value())
        self.synth_model_dim    = int(self.synth_model_dim_selection.currentText())
        self.FID_eval_freq      = int(self.FID_eval_Freq_slider.value() )
        self.fid_steps          = int(self.fid_steps_val_slider.value()) 
        self.z_train_gan_dim    = int(self.training_z_space_slider.value() )
        self.z_synth_gan_dim    = int(self.synth_z_space_slider.value())
        self.model_id           = self.model_id_box.toPlainText().strip()
        self.gan_output_type    = self.gan_output_image_type.currentText()
        self.net_gan_iter       = 0
        self.g_model_synthesis    = None
        self.syth_intensity     = bool(self.gan_include_intensit_img_flag.isChecked())
        #===========================     
        self.warn_status        = bool(self.Ignore_warning.isChecked())
        self.hflip_flag         = bool(self.rnd_flip_flag.isChecked())
        self.show_model_summary = bool(self.model_summary_flag.isChecked())
        self.rnd_img_ehnance    = bool(self.gan_rnd_enhance_flag.isChecked())
        self.rnd_img_brightness = bool(self.gan_rnd_brightness_flag.isChecked())
        self.rnd_img_contrast   = bool(self.gan_rnd_contrast_flag.isChecked())
        self.agument_sharpened  = bool(self.gan_sharpen_flag.isChecked())
        self.agument_blurred    = bool(self.gan_blurred_flag.isChecked())
        self.agument_denoised   = bool(self.gan_denoise_flag.isChecked())
        self.agument_distortion = bool(self.gan_image_distortion_flag.isChecked())
        self.agument_noise      = bool(self.gan_add_noise_flag.isChecked())
        self.agument_gaussian   = bool(self.gan_agu_gaussian_noise.isChecked()) 
        self.agument_random     = bool(self.gan_agu_random_noise.isChecked()) 
        self.agument_dilated_img= bool(self.gan_dilate_img_flag.isChecked())
        self.agument_eroded_img = bool(self.gan_erode_img_flag.isChecked())
        self.extract_using_roi  = bool(self.gan_set_roi_flag.isChecked())
        self.store_processed_img= bool(self.gan_store_processed_image_flag.isChecked())
        self.auto_roi_para_flag = bool(self.auto_set_roi_parameter_flag.isChecked())
        self.show_scan          = bool(self.show_scan_roi_flag.isChecked())
        self.store_roi_graph    = bool(self.flag_roi_graph_store.isChecked())
        self.watermark_imgs     = bool(self.img_mark_noise_flag.isChecked())
        self.preview_auto_roi   = bool(self.show_computed_roi.isChecked())
        self.Eval_FID_flag      = bool(self.FID_eval_flag.isChecked())
        self.gen_lrlu_flag      = bool(self.gen_lrelu_flag.isChecked())
        self.gen_lrelu_alpha    = round(self.gen_lrlu_alpha.value() , 3)
        self.dis_lrlu_flag      = bool(self.dis_lrelu_flag.isChecked())
        self.dis_lrelu_alpha    = round(self.dis_lrlu_alpha.value() , 3)
        self.gan_lr             = round(self.gmodel_lr.value(),7)
        self.gmodel_optimizer   = self.gan_optimizer.currentText().strip()
        self.gan_agument_flag   = bool(self.gan_img_agument_flag.isChecked())

        #=======================
        self.image = QImage()       
        #======Reset values
        self.cur_epoch = 0
        self.DRloss_list =[]
        self.DRloss_val = 0
        self.G_loss_list =[]
        self.G_loss_val = 0
        self.FID_list =[]
        self.Images =[]
        self.count_agumented = 0
        self.progressBar.setValue(0)
        #=============================Local class gloabl vaeiables
        self.win_start =0
        self.win_end   =0
        self.max_diff  =0
        self.Contour_classA =None                                            # for image A original during  dwaring rectangle
        self.Contour_classB =None                                            # for image B original during drawing rectangle
        self.deep_stack = []                                                 # for storing deepstack array
        self.multiclass_deep_stack = []                                      # for storing deepstack array
        self.ssim_list_data  = []
        self.multiclass_ssim_list_data  = []
        self.store_roi_graph    = bool(self.flag_roi_graph_store.isChecked())
        self.update_gan_parameters()                                        
        self.set_updated_gui_opts()                                           # do not use self.roi_dims here inside
        #============================
        self.time_stamp =  time_stamp()                                      # process time stamp

        #===========================                                         # Training Classifier variables
        self.class_model_id   = self.classifier_model_id.toPlainText().strip()
        self.cls_train_data   = self.classifier_source_path.toPlainText()
        self.cls_model_path   = self.classifier_model_path.toPlainText()
        self.cls_testing_path = self.classifier_testing_path.toPlainText()
        self.cls_output       = self.classifier_output_path.toPlainText()
        self.cls_process_type = self.cls_select_processor.currentText().strip()
        self.cls_tf_model_type= self.cls_tf_model.currentText().strip()
        self.cls_train_epoch  = self.classifier_traning_epoch.value()
        self.cls_autosave_freq= self.classifier_model_autosave.value()
        self.cls_batch_size   = self.classifier_batch_size_slider.value()
        self.show_cls_summary     = bool(self.cls_model_summary.isChecked())
        self.ignore_cls_warnings  = bool(self.cls_warning_flags.isChecked())
        self.cls_input_dim        = int(self.cls_input_shape_value.currentText())
        self.cls_lr               = round(self.cls_init_lr_val.value(), 10)
        self.cls_lr_decay_freq    = int(self.cls_lr_decay_slider.value())
        self.cls_lr_decay_rate    = int(self.cls_lr_decay_value.value())
        self.cls_optimizer        = self.cls_optimizer_type.currentText().strip()
        self.cls_lr_decay_flag    = bool(self.class_lr_freq_flag.isChecked())
        self.popup_win_flag       = bool(self.show_popup_win_flag.isChecked())
        self.use_offline_model    = bool(self.cls_use_offline_flag.isChecked())
        self.net_cls_epoch         = 0

        self.net_cls_loss         = []
        self.cur_cls_loss         = 0
        self.cur_cls_acc          = 0
        self.cls_accuracy         = []

        self.cls_val_accuracy     = []
        self.cur_cls_val_acc      = 0 
        self.net_cls_val_loss     = [] 
        self.cur_cls_val_loss     = 0
        self.net_auroc            = []
        self.cls_auroc            = []
        self.val_auroc            = []
        self.cur_val_auroc        = 0
        self.cur_cls_auroc        = 0


        #=========================================================================== Memory growth options
        self.gan_gpu_mem_growth   = bool(self.gan_mem_growth_flag.isChecked())
        self.cls_gpu_mem_growth   = bool(self.cls_mem_growth_flag.isChecked())
        #===========================================================================


    def show_info(self):
    	#=====================
        self.para_info  =  ""
        self.para_info += ("\n------------------[ CRISPs parameter ]------------------\n")
        self.para_info += ("\n#Process time stamp : %s"%self.time_stamp)
        self.para_info += ("\n#CRISP running mode : %s"%self.run_mode )
        self.para_info += ("\n#CRISP Logging mode : %s"%self.write_logfile  )

        if self.run_mode == "Extract_ROIs":
	       	self.para_info += ("\n\n==================[  Region of Interest   ]==================\n")
        	self.para_info += ("\n#CRISP running mode : %s"%self.run_mode )
        	self.para_info += ("\n#Extraction using ROI    : %s"%self.extract_using_roi)
        	if roi_dims != None:
        		self.para_info += ("\n#ROI Dims: (y1,y2,x1,x2) : %d,%d,%d,%d"%(roi_dims[0],roi_dims[1],roi_dims[2],roi_dims[3]))
        	self.para_info += ("\n#Process time stamp      : %s"%self.time_stamp)
        	self.para_info += ("\n#Contour imageA path     : %s"%self.cimageA_path )
        	self.para_info += ("\n#Contour imageB path     : %s"%self.cimageB_path )
        	self.para_info += ("\n#Gradient Threshold value: %s"%self.grad_threshold)
        	self.para_info += ("\n#Scan window size        : %d"%self.winsize      )
        	self.para_info += ("\n#No. of scan size        : %d"%self.win_segment )
        	self.para_info += ("\n#Scan windows shift size : %d"%self.win_shift )
        	self.para_info += ("\n#SSIM scoring method     : %s"%self.ssim_mode )      
        	self.para_info += ("\n#Scan SSIM thresthold    : %0.2f"%self.ssim_thresthold )
        	self.para_info += ("\n#Use Deep stacking       : %s"%self.stacking_flag )
        	self.para_info += ("\n#IAFG image FPS          : %d"%self.iafg_fps )
        	self.para_info += ("\n#IAFG auto FPS           : %s"%self.auto_fps)
        	self.para_info += ("\n#IAFG auto parameters    : %s"%self.auto_duration)
        	self.para_info += ("\n#IAFG cyclic duration(s) : %d"%self.iafg_duration)
        	self.para_info += ("\n#IAFG pre-process sharpen: %s"%self.iafg_sharpen)
        	self.para_info += ("\n#IAFG image weights      : %f"%self.iafg_weight)
        	self.para_info += ("\n#View IAFG processing    : %s"%self.iafg_preview)
        	self.para_info += ("\n#IAFG image rnd. select  : %s"%self.iafg_rnd_select)
        	self.para_info += ("\n#Output image format     : %s"%self.gan_output_type )

        if self.run_mode == "train_gan" or self.run_mode == "synthesis":
        	self.para_info += ("\n==================[  Generator | Synthesizer  ]==================\n")
        	self.para_info += ("\n#Model Name              : %s"%self.model_id)

        	if self.run_mode == "synthesis":
                    self.para_info += ("\n#Model output dimension  : (%d x %d)"%(self.synth_model_dim,self.synth_model_dim))
        	else:
                    self.para_info += ("\n#Model input dimension   : (%d x %d)"%(self.output_shape_value,self.output_shape_value))
                    
        	self.para_info += ("\n#GAN training mode       : %s"%self.gan_train_type) #self.gan_train_type
        	self.para_info += ("\n#Model path location     : %s"%self.stored_model_pth )
        	self.para_info += ("\n#Preview image location  : %s"%self.preview_imgs)
        	self.para_info += ("\n#Generated image location: %s"%self.gen_imgs_path)
        	self.para_info += ("\n#Store processed images  : %s"%self.processed_img_path )
        	self.para_info += ("\n#Source image path       : %s"%self.source_image_path ) 
        	self.para_info += ("\n#GAN traning mode        : %s"% self.run_mode  ) 
        	self.para_info += ("\n#GAN model optimizer     : %s"% self.gmodel_optimizer  ) 
        	self.para_info += ("\n#GAN model optimizer LR. : %f"% self.gan_lr  ) 
        	self.para_info += ("\n#Randomize GRED mode     : %s"%self.randomize_grad)   
        	self.para_info += ("\n#Gradient mode kernel    : %d x %d"%(self.grad_kernel_size, self.grad_kernel_size))  
        	self.para_info += ("\n#Total training iteration: %s"%self.traning_iter)
        	self.para_info += ("\n#Model autosave internal : %s"%self.autosave_freq )
        	self.para_info += ("\n#Batch size of training  : %s"%self.batch_size)
        	self.para_info += ("\n#Each output shape       : %s"%self.output_shape_value)
        	self.para_info += ("\n#Image Number in preview : %s"%self.img_preview )
        	self.para_info += ("\n#Evaluate FID for images : %s"%self.Eval_FID_flag)
        	self.para_info += ("\n#Z-dim vector (traning)  : %d"%self.z_dim)
        	self.para_info += ("\n#Synthetic intensity fac : (0,%d)"%self.gan_int_synth_fac)
        	self.para_info += ("\n#Z-dim vector (synthesis): %d"%self.z_dim)                     
        	self.para_info += ("\n#QScore Enabled          : %s"%self.blur_smp_flag)                     
        	self.para_info += ("\n#FID evaluation frequency: %d"%self.FID_eval_freq)                      
        	self.para_info += ("\n#Steps for FID calculate : %d"%self.fid_steps)
        	self.para_info += ("\n#Images to synthesize    : %d"%self.synth_img_number)            
        	self.para_info += ("\n#Steps for FID calculate : %d"%self.fid_steps)
        	self.para_info += ("\n#Output image grid dim.  : (%d x %d)"% (self.gan_img_grid_dim , self.gan_img_grid_dim ) )
        	self.para_info += ("\n#Sample image grid dim.  : (%d x %d)"% (self.sample_grid_dim  , self.sample_grid_dim ) )
        	self.para_info += ("\n#Preview image frequency : %s"%self.prev_img_freq)
        	self.para_info += ("\n#Use L-RELU in gen.      : %s"%self.gen_lrlu_flag)
        	self.para_info += ("\n#L-RELU alpha value      : %s"%self.gen_lrelu_alpha)
        	self.para_info += ("\n#Use L-RELU in discrim.  : %s"%self.dis_lrlu_flag)
        	self.para_info += ("\n#L-RELU alpha value      : %s"%self.dis_lrelu_alpha)
        	self.para_info += ("\n#Agument H.flipped image : %s"%self.hflip_flag )
        	self.para_info += ("\n#Agument sharpened image : %s"%self.agument_sharpened)
        	self.para_info += ("\n#Agument blurred images  : %s"%self.agument_blurred)
        	self.para_info += ("\n#Agument denoised  image : %s"%self.agument_denoised)
        	self.para_info += ("\n#Agument image distortion: %s"%self.agument_distortion)
        	self.para_info += ("\n#Agument enhanced edges  : %s"%self.rnd_img_ehnance)
        	self.para_info += ("\n#Agument rand. brightness: %s"%self.rnd_img_brightness)
        	self.para_info += ("\n#Agument rand. contrast  : %s"%self.rnd_img_contrast )
        	self.para_info += ("\n#Agument nosied image    : %s"%self.agument_noise)
        	self.para_info += ("\n#Agument dilated images  : %s"%self.agument_dilated_img )
        	self.para_info += ("\n#Dilation kernel, iter.  : (%d,%d), %d"%( self.img_dilate_kernel, self.img_dilate_kernel,  self.img_dilate_iter))       
        	self.para_info += ("\n#Agument eroded  images  : %s"%self.agument_eroded_img )
        	self.para_info += ("\n#Erosion kernel, iter.   : (%d,%d), %d"%(self.img_erosion_kernel, self.img_erosion_kernel, self.img_erosion_iter))
        	self.para_info += ("\n#Denoising kernel value  : %d"%self.denois_val)
        	self.para_info += ("\n#Sharpening kernel value : %d"%self.sharpen_val)   
        	self.para_info += ("\n#Agument Synthetic images: %s"%self.gan_agument_flag )
        	self.para_info += ("\n#Using Process type      : %s"%self.processor_type)
        	self.para_info += ("\n#Show model summary      : %s"%self.show_model_summary)
        	self.para_info += ("\n#Warning messages status : %s"%self.warn_status )

        if self.run_mode == "classifier_training"  or  self.run_mode ==  "classifier_inference" :
        	self.para_info += ("\n\n==================[ Classifier parameters ]==================\n")
        	self.para_info += ("\n#Class. model id         : %s"%self.class_model_id )
        	self.para_info += ("\n#Class. CNN model artict.: %s"%self.cls_tf_model_type)
        	self.para_info += ("\n#Class. training dim.    : (%d, %d)"%(self.cls_input_dim,self.cls_input_dim) )
        	self.para_info += ("\n#Class. training  path   : %s"%self.cls_train_data)
        	self.para_info += ("\n#Class. model path       : %s"%self.cls_model_path)
        	self.para_info += ("\n#Class. images path      : %s"%self.cls_testing_path) 
        	self.para_info += ("\n#Class. output path      : %s"%self.cls_output )
        	self.para_info += ("\n#Class. training batch   : %d"%self.cls_batch_size)
        	self.para_info += ("\n#Class. autosave freq.   : %d"%self.cls_autosave_freq)
        	self.para_info += ("\n#Class. initial LR       : %0.10f"%self.cls_lr)
        	self.para_info += ("\n#Class. LR decay enabled : %s"%self.cls_lr_decay_flag)
        	self.para_info += ("\n#Class. LR decay freq.   : %d"%self.cls_lr_decay_freq)
        	self.para_info += ("\n#Class. LR decay rate    : %d%%"%self.cls_lr_decay_rate)
        	self.para_info += ("\n#Class. training epoch   : %d"%self.cls_train_epoch  )             
       		self.para_info += ("\n#Class. optimizer type   : %s"%self.cls_optimizer    )
       		self.para_info += ("\n#Class. thresthold       : %0.2f"%self.inference_thrshold )
        	self.para_info += ("\n#Class. processor use    : %s"%self.cls_process_type ) 
        	self.para_info += ("\n#Class. memory growth    : %s"%self.gan_gpu_mem_growth )
        	if self.use_offline_model:
                    self.para_info += ("\n#Using Offline base model  : %s"%(self.use_offline_model))        	

    	#==========================================================================================
        self.para_info += ("\n================================================================\n")
        self.logger(self.para_info , color = 1)
    	#==========================================================================================                               # shows informations


    def save_custom_config(self):                                                                                                # store custom configuration to file
        if self.qm.question(self,'CRISP',"Save configuration to custom file?" , self.qm.Yes | self.qm.No) == self.qm.No:
            return

        options     = QFileDialog.Options()
        options    |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save Custom option settings","./config/","All Files (*.conf)", options=options)
        if fileName:
            if not fileName.lower().endswith(".conf"):
                fileName += ".conf"
            self.logger("[%s] | Custom configuration filename: %s"%(time_stamp(),fileName) )
            self.config_store(config_fname =  fileName)
        else:
            self.logger("[%s] | Warning! Custom configuration storage skipped by user..."%time_stamp())


    def config_store(self, config_fname = False):          # must put Flsaase here, None will not work

    	if config_fname == False:                                                                              # only ask when  save setting is pressed
    		if self.qm.question(self,'CouturGAN',"Confrim configuration save ?", self.qm.Yes | self.qm.No) == self.qm.No:
    			return

    	config = configparser.ConfigParser()
    	settings_file_path = "./config/settings.conf" if config_fname  == False else config_fname

    	config['GLOBAL'] = { "actionActivate_log"                   : self.actionActivate_log.isChecked(),  
    	                      "actionRestore_last_session_on_start" : self.actionRestore_last_session_on_start.isChecked() }


    	config['ROI']    = { 
                              #===========================|General ROI seelction GUI or cocordinates
    	                      "gan_set_roi_flag"      	        : self.gan_set_roi_flag.isChecked(),
    	                      "roi_from_gui"                    : self.roi_from_gui.isChecked(),
    	                      "roi_from_cord_flag"              : self.roi_from_cord_flag.isChecked(),
    	                      "sel_r1_start_time_slider"        : self.sel_r1_start_time_slider.value(),
    	                      "sel_r1_end_time_slider"          : self.sel_r1_end_time_slider.value(),
    	                      "sel_r2_start_time_slider"        : self.sel_r2_start_time_slider.value(),
    	                      "sel_r2_end_time_slider"          : self.sel_r2_end_time_slider.value(),
    	                      "orgdim_extraction_mode"          : self.orgdim_extraction_mode.currentIndex(),
    	                      "resize_orgdim_flag"              : self.resize_orgdim_flag.isChecked(),
    	                      "orgdim_width"                    : self.orgdim_width.value(),
    	                      "orgdim_height"                   : self.orgdim_height.value(),
    	                      "orgdim_output_type"              : self.orgdim_output_type.currentIndex(),
    	                      "roi_template_image_path"         : self.roi_template_image_path.toPlainText(),
    	                      "org_dim_rois_extract"            : self.org_dim_rois_extract.toPlainText(),
    	                      "lock_roi_val_flag"               : self.lock_roi_val_flag.isChecked(),
                              "roi_ds_id"                       : self.roi_ds_id.toPlainText(),
                              "roids_src_path"                  : self.roids_src_path.toPlainText(),   
                              "grad_threshold_value"            : self.grad_threshold_value.value(),
                              "flag_clone_multiclass_dataset"   : self.flag_clone_multiclass_dataset.isChecked(),  
                              "flag_pause_each_preprocessing"   : self.flag_pause_each_preprocessing.isChecked(),
                              "flag_autoset_multiclass_dataset_path" : self.flag_autoset_multiclass_dataset_path.isChecked(),  
                              #=====================================================================
                              "show_adaptive_trendline_flag"    : self.show_adaptive_trendline_flag.isChecked(),
                              "multiclass_val_cohort_flag"      : self.multiclass_val_cohort_flag.isChecked(),
                              "adaptive_colormap"               : self.adaptive_colormap.currentIndex(),
                              "adaptive_sort"                   : self.adaptive_sort.currentIndex(),
                              "adaptive_threshold_value"        : self.adaptive_threshold_value.value(),
                              "adaptive_kernel_value"           : self.adaptive_kernel_value.value(),
                              "adaptive_suffle_seed"            : self.adaptive_suffle_seed.value(),
                              "multiclass_source_dataset"       : self.multiclass_source_dataset.toPlainText(),
                              "multiclass_val_cohort"           : self.multiclass_val_cohort.toPlainText(), 
                              "multiclass_val_cohort_path"      : self.multiclass_val_cohort_path.toPlainText(), 
    	                      #==========================| IAFR contruction
    	                      "contourA_imagepath"              : self.contourA_imagepath.toPlainText(),
    	                      "contourB_imagepath"              : self.contourB_imagepath.toPlainText(),
    	                      "iafg_weight_val"                 : self.iafg_weight_val.value(),
    	                      "multiclass_iafg_weight_val"      : self.multiclass_iafg_weight_val.value(),
    	                      "iafg_duration_flag_val"          : self.iafg_duration_flag_val.isChecked(),
    	                      "multiclass_iafg_duration_flag_val" : self.multiclass_iafg_duration_flag_val.isChecked(),
    	                      "iafg_duration_slider"            : self.iafg_duration_slider.value(),
    	                      "multiclass_iafg_duration_slider" : self.multiclass_iafg_duration_slider.value(),
    	                      "multiclass_iafg_fps_flag_val"    : self.multiclass_iafg_fps_flag_val.isChecked(),
                              "iafg_fps_flag_val"               : self.iafg_fps_flag_val.isChecked(),
    	                      "iafg_fps_slider"                 : self.iafg_fps_slider.value(),
    	                      "multiclass_iafg_fps_slider"      : self.multiclass_iafg_fps_slider.value(),
    	                      "iafg_sharpen_flag_val"           : self.iafg_sharpen_flag_val.isChecked(),
    	                      "multiclass_iafg_sharpen_flag_val": self.multiclass_iafg_sharpen_flag_val.isChecked(),
    	                      "iafg_preview_flag_val"           : self.iafg_preview_flag_val.isChecked(),
    	                      "multiclass_iafg_preview_flag_val": self.multiclass_iafg_preview_flag_val.isChecked(),
    	                      "resize_iagf_output"              : self.resize_iagf_output.isChecked(),
    	                      "multiclass_resize_iagf_output"   : self.multiclass_resize_iagf_output.isChecked(),
    	                      "iagf_width"                      : self.iagf_width.value(),
    	                      "multiclass_iagf_width"           : self.multiclass_iagf_width.value(),
    	                      "iagf_height"                     : self.iagf_height.value(),
    	                      "multiclass_iagf_height"          : self.multiclass_iagf_height.value(),
    	                      "iafr_dilation_flag"              : self.iafr_dilation_flag.isChecked(),
    	                      "multiclass_iafr_dilation_flag"   : self.multiclass_iafr_dilation_flag.isChecked(),
    	                      "iafr_erosion_flag"               : self.iafr_erosion_flag.isChecked(),
    	                      "multiclass_iafr_erosion_flag"    : self.multiclass_iafr_erosion_flag.isChecked(),
    	                      "iafr_noise_flag"                 : self.iafr_noise_flag.isChecked(),
    	                      "multiclass_iafr_noise_flag"      : self.multiclass_iafr_noise_flag.isChecked(),
    	                      "iafr_denoise_flag"               : self.iafr_denoise_flag.isChecked(),
    	                      "multiclass_iafr_denoise_flag"    : self.multiclass_iafr_denoise_flag.isChecked(),
                              "flag_multiclass_pause_classes"   : self.flag_multiclass_pause_classes.isChecked(),  
                              "flag_pause_each_afrc"            : self.flag_pause_each_afrc.isChecked(),  
    	                      #==========================| AutoROIs & DeepStacking
    	                      "SSIM_mode_val"                   : self.SSIM_mode_val.currentIndex(),
    	                      "contourA_dspath"                 : self.contourA_dspath.toPlainText(),
    	                      "contourB_dspath"                 : self.contourB_dspath.toPlainText(),
    	                      "contourA_dspath_ds"              : self.contourA_dspath_ds.toPlainText(),
    	                      "contourB_dspath_ds"              : self.contourB_dspath_ds.toPlainText(),
    	                      "scan_winsize_slider"             : self.scan_winsize_slider.value(),
    	                      "window_shift_slider"             : self.window_shift_slider.value(),
    	                      "ssim_thresthold_slider"          : self.ssim_thresthold_slider.value(),
    	                      "scan_segment_slider"             : self.scan_segment_slider.value(),
                              "auto_set_roi_parameter_flag"     : self.auto_set_roi_parameter_flag.isChecked(),
                              "ds_pos_filepath"                 : self.ds_pos_filepath.toPlainText(),
    	                      #==========================| pre-processing & deepstacking
    	                      "sharpen_autoroi_flag"            : self.sharpen_autoroi_flag.isChecked(),
    	                      "denoise_autoroi_flag"            : self.denoise_autoroi_flag.isChecked(),
    	                      "use_validation_cohot_scores_flag": self.use_validation_cohot_scores_flag.isChecked(),                              
 	                        }



    	config['GAN']    = { 
                              #===========================|General source data and output paths
    	                      "gan_src_image"      	            : self.gan_src_image.toPlainText(),
    	                      "gan_model_path"              	: self.gan_model_path.toPlainText(), 
    	                      "gan_output_path"             	: self.gan_output_path.toPlainText(),
    	                      "preview_imgs_path"           	: self.preview_imgs_path.toPlainText(),  
    	                      "gan_train_mode"  	        	: self.gan_train_mode.currentIndex(),
    	                      "gan_store_processed_image_flag"	: self.gan_store_processed_image_flag.isChecked(),
    	                      "model_id_box"                	: self.model_id_box.toPlainText(),
    	                      #==========================| General Hyperparameters
    	                      "gan_processed_img_path"      	: self.gan_processed_img_path.toPlainText(),
    	                      "epochNum_slider"             	: self.epochNum_slider.value(),
    	                      "autosave_epoch"              	: self.autosave_epoch.value(),   
    	                      "FID_eval_flag"                   : self.FID_eval_flag.isChecked(), 
    	                      "FID_eval_Freq_slider"  	 	: self.FID_eval_Freq_slider.value(),
    	                      "set_batch"                   	: self.set_batch.currentIndex(),
    	                      "gan_output_image_type"       	: self.gan_output_image_type.currentIndex(),
    	                      "Sim_show_freq" 	            	: self.Sim_show_freq.value(),
    	                      "preview_num"                 	: self.preview_num.currentIndex(),
    	                      "preview_img_freq"            	: self.preview_img_freq.currentIndex(),
    	                      "gan_sample_dim"              	: self.gan_sample_dim.currentIndex(),
    	                      "gan_unit_shape"              	: self.gan_unit_shape.currentIndex(),
    	                      "img_mark_noise_flag"         	: self.img_mark_noise_flag.isChecked(),
    	                      #============================| Advance gan parameters

    	                      "training_z_space_slider"         : self.training_z_space_slider.value(),
    	                      "fid_steps_val_slider"            : self.fid_steps_val_slider.value(),
    	                      "gan_optimizer"                   : self.gan_optimizer.currentIndex() ,
    	                      "gmodel_lr"                       : self.gmodel_lr.value(),
    	                      "blur_sampling_flag"              : self.blur_sampling_flag.isChecked(),
    	                      "gen_lrelu_flag"                  : self.gen_lrelu_flag.isChecked(),
    	                      "gen_lrlu_alpha"                  : self.gen_lrlu_alpha.value(),
    	                      "dis_lrelu_flag"                  : self.dis_lrelu_flag.isChecked(),
    	                      "dis_lrlu_alpha"                  : self.dis_lrlu_alpha.value(),
    	                      #============================| Image agumentation filters
    	                      "gan_add_noise_flag"              : self.gan_add_noise_flag.isChecked(),
    	                      "gan_agu_random_noise"            : self.gan_agu_random_noise.isChecked(),
    	                      "gan_agu_gaussian_noise"          : self.gan_agu_gaussian_noise.isChecked(),
    	                      "gan_noise_intenisty"             : self.gan_noise_intenisty.value(),
    	                      "gan_sharpen_flag"                : self.gan_sharpen_flag.isChecked(),
    	                      "sharpen_kernel_slider"           : self.sharpen_kernel_slider.value(),
    	                      "gan_blurred_flag"                : self.gan_blurred_flag.isChecked(),
    	                      "blur_kernel_slider"              : self.blur_kernel_slider.value(),
    	                      "gan_denoise_flag"                : self.denoise_kernel_slider.value(),
    	                      "denoise_kernel_slider"           : self.denoise_kernel_slider.value(),
    	                      "gan_image_distortion_flag"       : self.gan_image_distortion_flag.isChecked(),
    	                      "gan_distortion_fac"              : self.gan_distortion_fac.value(),
    	                      "gan_dilate_img_flag"             : self.gan_dilate_img_flag.isChecked(),
    	                      "gan_dilation_kernel_slider"      : self.gan_dilation_kernel_slider.value(),
    	                      "gan_dilation_iter_slider"        : self.gan_dilation_iter_slider.value(),
    	                      "gan_erode_img_flag"              : self.gan_erode_img_flag.isChecked(),
    	                      "gan_erosion_kernel_slider"       : self.gan_erosion_kernel_slider.value(),
    	                      "gan_erosion_iter_slider"         : self.gan_erosion_iter_slider.value(),
                              "grad_randomize_flag"             : self.grad_randomize_flag.isChecked(),                              
    	                      "gradient_mode_slider_value"      : self.gradient_mode_slider_value.value(),
    	                      "rnd_flip_flag"                   : self.rnd_flip_flag.isChecked(),
    	                      "gan_rnd_contrast_flag"           : self.gan_rnd_contrast_flag.isChecked(),
    	                      "gan_rnd_brightness_flag"         : self.gan_rnd_brightness_flag.isChecked(),
    	                      "gan_rnd_enhance_flag"            : self.gan_rnd_enhance_flag.isChecked(),
    	                      #=============================| Synthesis parameters for GAN
    	                      "synth_model_dim_selection"       : self.synth_model_dim_selection.currentIndex(),
    	                      "gan_image_grid_val"              : self.gan_image_grid_val.currentIndex(),
    	                      "synth_img_no_slider"             : self.synth_img_no_slider.value(),
    	                      "synth_z_space_slider"            : self.synth_z_space_slider.value(),
    	                      "gan_intensity_fac"               : self.gan_intensity_fac.value(),
    	                      "gan_img_agument_flag"            : self.gan_img_agument_flag.isChecked(),
    	                      #=============================| System settings
    	                      "Ignore_warning"                  : self.Ignore_warning.isChecked(),
    	                      "model_summary_flag"              : self.model_summary_flag.isChecked(),
    	                      "SelectProcessor"                 : self.SelectProcessor.currentIndex()

    	                   }


    	config['CLS']    = { 
                              #===========================|Resoultion selction general paths
    	                      "classifier_source_path"      	: self.classifier_source_path.toPlainText(),
    	                      "classifier_model_path"      	    : self.classifier_model_path.toPlainText(),
    	                      "classifier_testing_path"         : self.classifier_testing_path.toPlainText(),    
    	                      "classifier_output_path"	        : self.classifier_output_path.toPlainText(),
    	                      "classifier_model_id"             : self.classifier_model_id.toPlainText(),
    	                      "cls_tf_model"                    : self.cls_tf_model.currentIndex(),
    	                      "cls_input_shape_value"           : self.cls_input_shape_value.currentIndex(),
    	                      #===========================| Training hyperparameters
    	                      "classifier_traning_epoch"        : self.classifier_traning_epoch.value(),
    	                      "classifier_batch_size_slider"    : self.classifier_batch_size_slider.value(),
    	                      "classifier_model_autosave"       : self.classifier_model_autosave.value(),
    	                      "cls_optimizer_type"              : self.cls_optimizer_type.currentIndex(),
    	                      "cls_init_lr_val"                 : self.cls_init_lr_val.value(),
    	                      "cls_lr_decay_slider"             : self.cls_lr_decay_slider.value(),
    	                      "cls_lr_decay_value"              : self.cls_lr_decay_value.value(),
    	                      #==========================| Training image agumentation
    	                      "cls_agu_hflip_flag"              : self.cls_agu_hflip_flag.isChecked(),
    	                      "cls_agu_vflip_flag"              : self.cls_agu_vflip_flag.isChecked(),
    	                      "cls_agu_img_shear_flag"          : self.cls_agu_img_shear_flag.isChecked(),
                              "cls_img_zoom_flag"               : self.cls_img_zoom_flag.isChecked(),                              
    	                      "cls_img_shear_slider"            : self.cls_img_shear_slider.value(),
    	                      "cls_img_zoom_slider"             : self.cls_img_zoom_slider.value(),
    	                      "cls_img_rot_slider"              : self.cls_img_rot_slider.value(),
    	                      #==========================| Classifier inferencing    	
    	                      "cls_prediction_level"            : self.cls_prediction_level.currentIndex(),
    	                      "btn_cls_show_pred_heatmap_flag"  : self.btn_cls_show_pred_heatmap_flag.isChecked(),
    	                      "flag_inference_include_subfolders"  : self.flag_inference_include_subfolders.isChecked(),                              
    	                      "flag_show_basic_inference_stats"    : self.flag_show_basic_inference_stats.isChecked(),   
    	                      "btn_pause_each_heatmap_flag"        : self.btn_pause_each_heatmap_flag.isChecked(),
    	                      "cls_store_pred_heatmap_path"        : self.cls_store_pred_heatmap_path.toPlainText(),
    	                      "cls_thrhld_slider"               : self.cls_thrhld_slider.value(),
    	                      "cls_output_type"                 : self.cls_output_type.currentIndex(),
    	                      "cls_write_output_flag"           : self.cls_write_output_flag.isChecked(),
    	                      "classifier_report_path"          : self.classifier_report_path.toPlainText(),
    	                      "cls_verbosity_level"             : self.cls_verbosity_level.currentIndex(),
    	                      "watermark_inference_flag"        : self.watermark_inference_flag.isChecked()

    	                     }

    	print("[%s] | Storing configuration to default file: %s"%(time_stamp(), settings_file_path) )

    	with open(settings_file_path, 'w') as configfile:                           # store configuration to setting.conf
            config.write(configfile)

    	self.logger("\n[%s] | Custom configuration stored succssefully...."%time_stamp())
    	
    	self.store_ssim_pos()                                                      # store for non muticlass and singlr ROIs
    	self.multiclass_store_ssim_pos()                                           # store the SSIM values, deepstacks & ....

    	self.logger("\n[%s] | DeepStacking & ROIs values were be stored if present...."%time_stamp())
    	#================================================================================================================================


    def store_ssim_pos(self):

    	if len(self.deep_stack) and len(self.ssim_list_data)> 0 :                                              # write the deepstak information to  single pair deepsdtacking data file
            ds_datafile = self.ds_pos_filepath.toPlainText()
            with open(ds_datafile, 'wb') as ssim_datafile:
                data_obj = [self.roi_dims, self.deep_stack, self.ssim_list_data]                               # single cass deepstacking dataset
                pickle.dump(data_obj , ssim_datafile)
                print("[%s] | Possible Deep Stacking ROIs & other serrings stored successfully..."%time_stamp())


    	if self.roi_dims != None:                                                                             # only for teh sub-case of extraction ROIs
            # store single ROi as well for extraction
            single_roi_fnam = os.path.join(self.roids_src_path.toPlainText() , self.roi_ds_id.toPlainText() +".data")
            with open(single_roi_fnam, 'wb') as single_ROI_datafile:
                data_obj = [self.roi_dims, self.roids_src_path.toPlainText(), self.roi_ds_id.toPlainText() ]  # co-ordinates, sourc path, class name
                pickle.dump(data_obj , single_ROI_datafile)
                print("[%s] | Single ROI data for extraction stroed successfully..."%time_stamp())

    	self.logger("\n[%s] | All configuration stored succssefully...."%time_stamp())
        #==============================================================================================================================


    def multiclass_store_ssim_pos(self):

        if len(self.multiclass_deep_stack) and len(self.multiclass_ssim_list_data)> 0 :                                              # writet he deepstak information to sepearte data file
            ds_datafile = os.path.join(os.path.dirname(self.multiclass_ds_pos_fpath.toPlainText()), "afrc_image", "rois_ds_info.data")
            with open(ds_datafile, 'wb') as ssim_datafile:
                data_obj = [self.multiclass_roi_dims, self.multiclass_deep_stack, self.multiclass_ssim_list_data]
                pickle.dump(data_obj , ssim_datafile)

        #==============================================================================================================================


    def load_default_config(self):
    	self.load_config(config_fname = "./config/default.conf",  msg = "Resetting to default configuration?")


    def load_model_config(self):
    	self.load_config(config_fname = "./config/settings.conf",  msg = "Load model configuration?")


    def load_config(self, config_fname = "./config/default.conf",  msg = "Load default configuration?"):

    	if args.config_run == 0 and msg !=  "":                                                                 # do not ask during config coomadn line run
            if self.qm.question(self,'CRISP-II', msg, self.qm.Yes | self.qm.No) == self.qm.No:
                return

    	self.logger ("\n[%s] | Loading Setting configurations..."%time_stamp())
    	config = configparser.ConfigParser() 	
    	config.read( config_fname )

    	# Selection ROI
    	self.actionRestore_last_session_on_start.setChecked(to_bool(config["GLOBAL"]["actionRestore_last_session_on_start"]) )   
    	self.actionActivate_log.setChecked( to_bool(config["GLOBAL"]["actionActivate_log"]) )

    	self.gan_set_roi_flag.setChecked( to_bool(config["ROI"]["gan_set_roi_flag"]) )
    	self.roi_from_gui.setChecked( to_bool(config["ROI"]["roi_from_gui"]) )
    	self.roi_from_cord_flag.setChecked( to_bool(config["ROI"]["roi_from_cord_flag"]) )
    	self.sel_r1_start_time_slider.setValue(config["ROI"].getfloat("sel_r1_start_time_slider") )
    	self.sel_r1_end_time_slider.setValue(config["ROI"].getfloat("sel_r1_end_time_slider") )
    	self.sel_r2_start_time_slider.setValue(config["ROI"].getfloat("sel_r2_start_time_slider") )
    	self.sel_r2_end_time_slider.setValue(config["ROI"].getfloat("sel_r2_end_time_slider") )
    	self.orgdim_extraction_mode.setCurrentIndex(config["ROI"].getint("orgdim_extraction_mode") )
    	self.resize_orgdim_flag.setChecked( to_bool(config["ROI"]["resize_orgdim_flag"]) )
    	self.orgdim_width.setValue(config["ROI"].getfloat("orgdim_width") )
    	self.orgdim_height.setValue(config["ROI"].getfloat("orgdim_height") )
    	self.orgdim_output_type.setCurrentIndex(config["ROI"].getint("orgdim_output_type") )
    	self.roi_template_image_path.setText(config["ROI"]["roi_template_image_path"] )
    	self.org_dim_rois_extract.setText(config["ROI"]["org_dim_rois_extract"] )
    	self.lock_roi_val_flag.setChecked( to_bool(config["ROI"]["lock_roi_val_flag"]) )
    	self.contourA_imagepath.setText(config["ROI"]["contourA_imagepath"] )
    	self.contourB_imagepath.setText(config["ROI"]["contourB_imagepath"] )
    	self.iafg_weight_val.setValue(config["ROI"].getfloat("iafg_weight_val") )
    	self.multiclass_source_dataset.setText(config["ROI"]["multiclass_source_dataset"])
    	self.multiclass_iafg_weight_val.setValue(config["ROI"].getfloat("multiclass_iafg_weight_val") )
    	self.iafg_duration_flag_val.setChecked( to_bool(config["ROI"]["iafg_duration_flag_val"]) )
    	self.multiclass_iafg_duration_flag_val.setChecked( to_bool(config["ROI"]["multiclass_iafg_duration_flag_val"]) )
    	self.iafg_duration_slider.setValue(config["ROI"].getfloat("iafg_duration_slider") )
    	self.multiclass_iafg_duration_slider.setValue(config["ROI"].getfloat("multiclass_iafg_duration_slider") )
    	self.iafg_fps_flag_val.setChecked( to_bool(config["ROI"]["iafg_fps_flag_val"]) )
    	self.multiclass_iafg_fps_flag_val.setChecked( to_bool(config["ROI"]["multiclass_iafg_fps_flag_val"]) )
    	self.iafg_fps_slider.setValue(config["ROI"].getfloat("iafg_fps_slider") )
    	self.multiclass_iafg_fps_slider.setValue(config["ROI"].getfloat("multiclass_iafg_fps_slider") )
    	self.iafg_sharpen_flag_val.setChecked( to_bool(config["ROI"]["iafg_sharpen_flag_val"]) )
    	self.multiclass_iafg_sharpen_flag_val.setChecked( to_bool(config["ROI"]["multiclass_iafg_sharpen_flag_val"]) )
    	self.iafg_preview_flag_val.setChecked( to_bool(config["ROI"]["iafg_preview_flag_val"]) )
    	self.multiclass_iafg_preview_flag_val.setChecked( to_bool(config["ROI"]["multiclass_iafg_preview_flag_val"]) )
    	self.resize_iagf_output.setChecked( to_bool(config["ROI"]["resize_iagf_output"]) )
    	self.multiclass_resize_iagf_output.setChecked( to_bool(config["ROI"]["multiclass_resize_iagf_output"]) )
    	self.iagf_width.setValue(config["ROI"].getfloat("iagf_width") )
    	self.multiclass_iagf_width.setValue(config["ROI"].getfloat("multiclass_iagf_width") )
    	self.iagf_height.setValue(config["ROI"].getfloat("iagf_height") )
    	self.multiclass_iagf_height.setValue(config["ROI"].getfloat("multiclass_iagf_height") )
    	self.iafr_dilation_flag.setChecked( to_bool(config["ROI"]["iafr_dilation_flag"]) )
    	self.multiclass_iafr_dilation_flag.setChecked( to_bool(config["ROI"]["multiclass_iafr_dilation_flag"]) )
    	self.iafr_erosion_flag.setChecked( to_bool(config["ROI"]["iafr_erosion_flag"]) )
    	self.multiclass_iafr_erosion_flag.setChecked( to_bool(config["ROI"]["multiclass_iafr_erosion_flag"]) )
    	self.iafr_noise_flag.setChecked( to_bool(config["ROI"]["iafr_noise_flag"]) )
    	self.multiclass_iafr_noise_flag.setChecked( to_bool(config["ROI"]["multiclass_iafr_noise_flag"]) )
    	self.iafr_denoise_flag.setChecked( to_bool(config["ROI"]["iafr_denoise_flag"]) )
    	self.multiclass_iafr_denoise_flag.setChecked( to_bool(config["ROI"]["multiclass_iafr_denoise_flag"]) )
    	self.SSIM_mode_val.setCurrentIndex(config["ROI"].getint("SSIM_mode_val") )
    	self.contourA_dspath.setText(config["ROI"]["contourA_dspath"] )
    	self.contourB_dspath.setText(config["ROI"]["contourB_dspath"] )
    	self.contourA_dspath_ds.setText(config["ROI"]["contourA_dspath_ds"] )
    	self.contourB_dspath_ds.setText(config["ROI"]["contourB_dspath_ds"] )
    	self.scan_winsize_slider.setValue(config["ROI"].getfloat("scan_winsize_slider") )
    	self.window_shift_slider.setValue(config["ROI"].getfloat("window_shift_slider") )
    	self.ssim_thresthold_slider.setValue(config["ROI"].getfloat("ssim_thresthold_slider") )
    	self.scan_segment_slider.setValue(config["ROI"].getfloat("scan_segment_slider") )
    	self.sharpen_autoroi_flag.setChecked( to_bool(config["ROI"]["sharpen_autoroi_flag"]) )
    	self.denoise_autoroi_flag.setChecked( to_bool(config["ROI"]["denoise_autoroi_flag"]) )
    	self.auto_set_roi_parameter_flag.setChecked( to_bool(config["ROI"]["auto_set_roi_parameter_flag"]) )
    	self.ds_pos_filepath.setText(config["ROI"]["ds_pos_filepath"] )
    	self.roids_src_path.setText(config["ROI"]["roids_src_path"])
    	self.roi_ds_id.setText(config["ROI"]["roi_ds_id"])
    	self.flag_multiclass_pause_classes.setChecked(to_bool(config["ROI"]["flag_multiclass_pause_classes"]))
    	self.flag_pause_each_afrc.setChecked(to_bool(config["ROI"]["flag_pause_each_afrc"]))
    	self.roi_imagepath = self.roi_template_image_path.toPlainText()
    	self.grad_threshold_value.setValue(config["ROI"].getint("grad_threshold_value") )
    	self.flag_clone_multiclass_dataset.setChecked(to_bool(config["ROI"]["flag_clone_multiclass_dataset"] ))  
    	self.flag_pause_each_preprocessing.setChecked(to_bool(config["ROI"]["flag_pause_each_preprocessing"] )) 
    	self.flag_autoset_multiclass_dataset_path.setChecked(to_bool(config["ROI"]["flag_autoset_multiclass_dataset_path"] ))   
    	self.multiclass_val_cohort_flag.setChecked(to_bool(config["ROI"]["multiclass_val_cohort_flag"] ))          
    	self.multiclass_val_cohort.setText(config["ROI"]["multiclass_val_cohort"] )
    	self.multiclass_val_cohort_path.setText(config["ROI"]["multiclass_val_cohort_path"] )
    	self.use_validation_cohot_scores_flag.setChecked(to_bool(config["ROI"]["use_validation_cohot_scores_flag"] ))
    	self.adaptive_colormap.setCurrentIndex(int(config["ROI"]["adaptive_colormap"]))
    	self.adaptive_sort.setCurrentIndex(int(config["ROI"]["adaptive_sort"]))
    	self.show_adaptive_trendline_flag.setChecked(to_bool(config["ROI"]["show_adaptive_trendline_flag"]))
    	self.adaptive_threshold_value.setValue(config["ROI"].getfloat("grad_threshold_value") )
    	self.adaptive_suffle_seed.setValue(config["ROI"].getfloat("adaptive_suffle_seed") )
    	self.adaptive_kernel_value.setValue(config["ROI"].getfloat("adaptive_kernel_value") )  

    	# Selection GAN
    	self.gan_src_image.setText(config["GAN"]["gan_src_image"] )
    	self.gan_model_path.setText(config["GAN"]["gan_model_path"] )
    	self.gan_output_path.setText(config["GAN"]["gan_output_path"] )
    	self.preview_imgs_path.setText(config["GAN"]["preview_imgs_path"] )
    	self.gan_train_mode.setCurrentIndex(config["GAN"].getint("gan_train_mode") )
    	self.gan_store_processed_image_flag.setChecked( to_bool(config["GAN"]["gan_store_processed_image_flag"]) )
    	self.model_id_box.setText(config["GAN"]["model_id_box"] )
    	self.gan_processed_img_path.setText(config["GAN"]["gan_processed_img_path"] )
    	self.epochNum_slider.setValue(config["GAN"].getfloat("epochNum_slider") )
    	self.blur_sampling_flag.setChecked( to_bool(config["GAN"]["blur_sampling_flag"]) )
    	self.autosave_epoch.setValue(config["GAN"].getfloat("autosave_epoch") )
    	self.FID_eval_flag.setChecked( to_bool(config["GAN"]["FID_eval_flag"]) )
    	self.FID_eval_Freq_slider.setValue(config["GAN"].getint("FID_eval_Freq_slider") )
    	self.set_batch.setCurrentIndex(config["GAN"].getint("set_batch") )
    	self.gan_output_image_type.setCurrentIndex(config["GAN"].getint("gan_output_image_type") )
    	self.Sim_show_freq.setValue(config["GAN"].getfloat("Sim_show_freq") )
    	self.preview_num.setCurrentIndex(config["GAN"].getint("preview_num") )
    	self.preview_img_freq.setCurrentIndex(config["GAN"].getint("preview_img_freq") )
    	self.gan_sample_dim.setCurrentIndex(config["GAN"].getint("gan_sample_dim") )
    	self.gan_unit_shape.setCurrentIndex(config["GAN"].getint("gan_unit_shape") )
    	self.img_mark_noise_flag.setChecked( to_bool(config["GAN"]["img_mark_noise_flag"]) )
    	self.training_z_space_slider.setValue(config["GAN"].getfloat("training_z_space_slider") )
    	self.fid_steps_val_slider.setValue(config["GAN"].getfloat("fid_steps_val_slider") )
    	self.gan_optimizer.setCurrentIndex(config["GAN"].getint("gan_optimizer") )
    	self.gmodel_lr.setValue(config["GAN"].getfloat("gmodel_lr") )
    	self.gen_lrelu_flag.setChecked( to_bool(config["GAN"]["gen_lrelu_flag"]) )
    	self.gen_lrlu_alpha.setValue(config["GAN"].getfloat("gen_lrlu_alpha") )
    	self.dis_lrelu_flag.setChecked( to_bool(config["GAN"]["dis_lrelu_flag"]) )
    	self.dis_lrlu_alpha.setValue(config["GAN"].getfloat("dis_lrlu_alpha") )
    	self.gan_add_noise_flag.setChecked( to_bool(config["GAN"]["gan_add_noise_flag"]) )
    	self.gan_agu_random_noise.setChecked( to_bool(config["GAN"]["gan_agu_random_noise"]) )
    	self.gan_agu_gaussian_noise.setChecked( to_bool(config["GAN"]["gan_agu_gaussian_noise"]) )
    	self.gan_noise_intenisty.setValue(config["GAN"].getfloat("gan_noise_intenisty") )
    	self.gan_sharpen_flag.setChecked( to_bool(config["GAN"]["gan_sharpen_flag"]) )
    	self.sharpen_kernel_slider.setValue(config["GAN"].getfloat("sharpen_kernel_slider") )
    	self.gan_blurred_flag.setChecked( to_bool(config["GAN"]["gan_blurred_flag"]) )
    	self.blur_kernel_slider.setValue(config["GAN"].getfloat("blur_kernel_slider") )
    	self.gan_denoise_flag.setChecked(config["GAN"].getfloat("gan_denoise_flag") )
    	self.denoise_kernel_slider.setValue(config["GAN"].getfloat("denoise_kernel_slider") )
    	self.gan_image_distortion_flag.setChecked( to_bool(config["GAN"]["gan_image_distortion_flag"]) )
    	self.gan_distortion_fac.setValue(config["GAN"].getfloat("gan_distortion_fac") )
    	self.gan_dilate_img_flag.setChecked( to_bool(config["GAN"]["gan_dilate_img_flag"]) )
    	self.gan_dilation_kernel_slider.setValue(config["GAN"].getfloat("gan_dilation_kernel_slider") )
    	self.gan_dilation_iter_slider.setValue(config["GAN"].getfloat("gan_dilation_iter_slider") )
    	self.gan_erode_img_flag.setChecked( to_bool(config["GAN"]["gan_erode_img_flag"]) )
    	self.gan_erosion_kernel_slider.setValue(config["GAN"].getfloat("gan_erosion_kernel_slider") )
    	self.gan_erosion_iter_slider.setValue(config["GAN"].getfloat("gan_erosion_iter_slider") )
    	self.grad_randomize_flag.setChecked( to_bool(config["GAN"]["grad_randomize_flag"]) )
    	self.gradient_mode_slider_value.setValue(config["GAN"].getfloat("gradient_mode_slider_value") )
    	self.rnd_flip_flag.setChecked( to_bool(config["GAN"]["rnd_flip_flag"]) )
    	self.gan_rnd_contrast_flag.setChecked( to_bool(config["GAN"]["gan_rnd_contrast_flag"]) )
    	self.gan_rnd_brightness_flag.setChecked( to_bool(config["GAN"]["gan_rnd_brightness_flag"]) )
    	self.gan_rnd_enhance_flag.setChecked( to_bool(config["GAN"]["gan_rnd_enhance_flag"]) )
    	self.synth_model_dim_selection.setCurrentIndex(config["GAN"].getint("synth_model_dim_selection") )
    	self.gan_image_grid_val.setCurrentIndex(config["GAN"].getint("gan_image_grid_val") )
    	self.synth_img_no_slider.setValue(config["GAN"].getfloat("synth_img_no_slider") )
    	self.synth_z_space_slider.setValue(config["GAN"].getfloat("synth_z_space_slider") )
    	self.gan_intensity_fac.setValue(config["GAN"].getfloat("gan_intensity_fac") )
    	self.gan_img_agument_flag.setChecked( to_bool(config["GAN"]["gan_img_agument_flag"]) )
    	self.Ignore_warning.setChecked( to_bool(config["GAN"]["Ignore_warning"]) )
    	self.model_summary_flag.setChecked( to_bool(config["GAN"]["model_summary_flag"]) )
    	self.SelectProcessor.setCurrentIndex(config["GAN"].getint("SelectProcessor") )

    	# Selection CLS
    	self.classifier_source_path.setText(config["CLS"]["classifier_source_path"] )
    	self.classifier_model_path.setText(config["CLS"]["classifier_model_path"] )
    	self.classifier_testing_path.setText(config["CLS"]["classifier_testing_path"] )
    	self.classifier_output_path.setText(config["CLS"]["classifier_output_path"] )
    	self.classifier_model_id.setText(config["CLS"]["classifier_model_id"] )
    	self.cls_tf_model.setCurrentIndex(config["CLS"].getint("cls_tf_model") )
    	self.cls_input_shape_value.setCurrentIndex(config["CLS"].getint("cls_input_shape_value") )
    	self.classifier_traning_epoch.setValue(config["CLS"].getfloat("classifier_traning_epoch") )
    	self.classifier_batch_size_slider.setValue(config["CLS"].getfloat("classifier_batch_size_slider") )
    	self.classifier_model_autosave.setValue(config["CLS"].getfloat("classifier_model_autosave") )
    	self.cls_optimizer_type.setCurrentIndex(config["CLS"].getint("cls_optimizer_type") )
    	self.cls_init_lr_val.setValue(config["CLS"].getfloat("cls_init_lr_val") )
    	self.cls_lr_decay_slider.setValue(config["CLS"].getfloat("cls_lr_decay_slider") )
    	self.cls_lr_decay_value.setValue(config["CLS"].getfloat("cls_lr_decay_value") )
    	self.cls_agu_hflip_flag.setChecked( to_bool(config["CLS"]["cls_agu_hflip_flag"]) )
    	self.cls_agu_vflip_flag.setChecked( to_bool(config["CLS"]["cls_agu_vflip_flag"]) )
    	self.cls_agu_img_shear_flag.setChecked( to_bool(config["CLS"]["cls_agu_img_shear_flag"]) )
    	self.cls_img_zoom_flag.setChecked( to_bool(config["CLS"]["cls_img_zoom_flag"]) )
    	self.cls_img_shear_slider.setValue(config["CLS"].getfloat("cls_img_shear_slider") )
    	self.cls_img_zoom_slider.setValue(config["CLS"].getfloat("cls_img_zoom_slider") )
    	self.cls_img_rot_slider.setValue(config["CLS"].getfloat("cls_img_rot_slider") )
    	self.cls_thrhld_slider.setValue(config["CLS"].getfloat("cls_thrhld_slider") )
    	self.cls_output_type.setCurrentIndex(config["CLS"].getint("cls_output_type") )
    	self.cls_write_output_flag.setChecked( to_bool(config["CLS"]["cls_write_output_flag"]) )
    	self.classifier_report_path.setText(config["CLS"]["classifier_report_path"] )
    	self.cls_verbosity_level.setCurrentIndex(config["CLS"].getint("cls_verbosity_level") )
    	self.watermark_inference_flag.setChecked( to_bool(config["CLS"]["watermark_inference_flag"]) )
        #Global options for the Logging
    	self.actionActivate_log.setChecked(to_bool(config["GLOBAL"]["actionActivate_log"]) )
    	self.cls_prediction_level.setCurrentIndex(config["CLS"].getint("cls_prediction_level") )
    	self.btn_cls_show_pred_heatmap_flag.setChecked( to_bool(config["CLS"]["btn_cls_show_pred_heatmap_flag"]) )
    	self.flag_inference_include_subfolders.setChecked( to_bool(config["CLS"]["flag_inference_include_subfolders"]) )
    	self.flag_show_basic_inference_stats.setChecked( to_bool(config["CLS"]["flag_show_basic_inference_stats"]) )  
    	self.btn_pause_each_heatmap_flag.setChecked( to_bool(config["CLS"]["btn_pause_each_heatmap_flag"]) )
    	self.cls_store_pred_heatmap_path.setText(config["CLS"]["cls_store_pred_heatmap_path"] )
        #======================================================== # read the values and update the SSIm list
    	self.load_ssim_data(show_msg = False)  if msg == "" else  self.load_ssim_data()                       # load the ROIs & deep stacking data
      

    def load_ssim_data(self , show_msg = True):
    	if show_msg:
            if self.qm.question(self,"CouturGAN", "Restore the last used single-pair DeepStacking and ROIs values for given configuration (if exists)?" , self.qm.Yes | self.qm.No) == self.qm.No:
                return

    	ds_datafile =self.ds_pos_filepath.toPlainText()
    	if os.path.isfile(ds_datafile):
            with open(ds_datafile, 'rb') as ssim_datafile:
                [self.roi_dims, self.deep_stack, self.ssim_list_data] = pickle.load(ssim_datafile)
                print("[%s] | ROIs & Deepstacking co-ordinates previously stored were... "%time_stamp())
                print("ROI Extraction co-ordinates :" , self.roi_dims  )
                print("DeepStack co-ordinates      :" , self.deep_stack)

            self.roi_imagepath = self.roi_template_image_path.toPlainText()                                   # restore the ROI extraction tempalte file name image path

            if len(self.ssim_list_data) > 0:
                self.SSIM_list.clear()
                for each_roi in self.ssim_list_data:
                    self.SSIM_list.addItem(each_roi)
                print("[%s] | DeepStacking co-ordinates and ROIs restored successfully..."%time_stamp())
            else:
                print("[%s] | Warning! No previous ROIs & DeepStacking Co-ordinates were found..."%time_stamp())
        #========================================================
        # single ROI data load        
    	single_roi_fnam = os.path.join(self.roids_src_path.toPlainText() , self.roi_ds_id.toPlainText() +".data")
    	with open(single_roi_fnam,  'rb') as single_ROI_datafile:
            self.roi_dims,_,_ = pickle.load(single_ROI_datafile)
            print("Stored co-ordiates for Single ROI: ", self.roi_dims )
            print("[%s] | Single ROI data for extraction stroed successfully..."%time_stamp())

    	self.logger("\n[%s] | All configuration stored succssefully...."%time_stamp())




    def multiclass_load_ssim_data(self, show_msg = True):

        if show_msg:
            if self.qm.question(self,"CouturGAN", "Restore the last multi-class used DeepStacking and ROIs values for given configuration (if exists)?" , self.qm.Yes | self.qm.No) == self.qm.No:
                return

        ds_datafile =self.multiclass_ds_pos_fpath.toPlainText()                                              # get the filepath for /rois_ds_info.data
        self.multiclass_ds_pos_dspath.setText(os.path.basename(os.path.dirname()))                           # get ethe class of the disease e.g.: HD_DM for ../HD_DM/rois_ds_info.data"

        if os.path.isfile(ds_datafile):
            with open(ds_datafile, 'rb') as ssim_datafile:
                [self.multiclass_roi_dims, self.multiclass_deep_stack, self.multiclass_ssim_list_data] = pickle.load(ssim_datafile)
                print("[%s] | ROIs & Deepstacking co-ordinates previously stored were... "%time_stamp())
                print("ROI:", self.multiclass_roi_dims)
                print("Deep stacking co-ordinates: ", self.multiclass_deep_stack)

            self.multiclass_roi_imagepath = self.multiclass_roi_template_image_path.toPlainText()           # restore the ROI extraction tempalte file name image path

            if len(self.multiclass_ssim_list_data) > 0:
                self.multiclass_SSIM_list.clear()
                for each_roi in self.multiclass_ssim_list_data:
                    self.multiclass_SSIM_list.addItem(each_roi)
                print("[%s] | Multi-class DeepStacking co-ordinates and ROIs restored (if exists)..."%time_stamp())
            else:
                print("[%s] | Warning! No previous ROIs & DeepStacking Co-ordinates were found..."%time_stamp())
        #========================================================


    def write_gan_summary(self, write_to_file = True):                                                        # Swarmplt for the blur fac 
        try:
            gan_model_info  = "|===================| Generator Model |=====================|\n"
            gan_model_info += "| \n"
            gan_model_info += "| Model name            : %s\n"%self.model_id
            gan_model_info += "| Model resolution      : (%s x %s) pixels\n"%(str(self.mode),str(self.mode))
            gan_model_info += "| Training data source  : %s\n"%self.source_image_path 
            gan_model_info += "| Model source type     : %s\n"%self.gan_train_type[ : self.gan_train_type.find("::") ]  
            gan_model_info += "| Model Z-dim vector    : %s\n"%str(self.z_train_gan_dim)
            gan_model_info += "| Model optimizer       : %s\n"%str(self.gmodel_optimizer)
            gan_model_info += "| Total Iteration       : %d\n"%self.net_gan_iter
            gan_model_info += "| Target Iteration      : %d\n"%self.traning_iter
            if len(self.FID_list) > 0:
                epch, value =  min(self.FID_list, key = lambda t: t[1])
                gan_model_info += "| Best FID (epoch,value):(%s, %s)\n"%(str(epch), str(value))
            else:
                gan_model_info += "| Best FID Value        : n/a\n"
            gan_model_info += "| Training batch size   : %d\n"%self.batch_size
            gan_model_info += "| Trained on            : %s\n"%self.processor_type
            gan_model_info += "| \n"
            gan_model_info += "|===========================================================|\n"

            summary_fname = "gan_model_" + self.model_id + "_" + str(self.mode) + "_model_summary.txt"

            if write_to_file:
                with open(self.stored_model_pth +"/"+ summary_fname , 'w') as model_summary:
                    model_summary.write(gan_model_info)

                with open(self.stored_model_pth +"/"+ self.model_id +".model" , 'w') as model_id:   # write model ID file
                	model_id.write("\nID configuration tag for GAN model :%s"%self.model_id)

                self.logger("\n[%s] | GAN model summary written successfully..."%time_stamp())

            else:
                self.logger( "\n" + gan_model_info,  color = 4)                                                   # only display the model before running
        except:
            self.logger("\n[%s] | Failed to display or write gan model summary due to unknown error..."%time_stamp(), color =5)
            pass




    def write_class_summary(self, write_to_file = True):                                 # display write to file 
        try:
            cls_model_info  = "|===================| Classifier Model |====================|\n"
            cls_model_info += "| \n"
            cls_model_info += "| Model name            : %s\n"%self.class_model_id
            cls_model_info += "| Model input dimension : (%s, %s) pixels\n"%( str(self.cls_input_dim), str(self.cls_input_dim) )
            cls_model_info += "| Training data source  : %s\n"%self.cls_train_data
            cls_model_info += "| Model architecture    : %s\n"%self.cls_tf_model_type
            cls_model_info += "| Total epoch(s) run    : %d\n"%self.net_cls_epoch
            cls_model_info += "| Target epoch(s) run   : %d\n"%self.cls_train_epoch
            cls_model_info += "| Learning rate (LR)    : %s\n"%str(self.cls_lr )
            if self.cls_lr_decay_flag :
                cls_model_info += "| LR Decay enabled      : True (%s%% per %s Iter.)\n"%(str(self.cls_lr_decay_rate), str(self.cls_lr_decay_freq)) 
            else:
                cls_model_info += "| LR Decay enabled      : False\n"
            try:
                cls_model_info += "| Model Kappa score     : %f\n"%str(self.kappa)
                cls_model_info += "| Model F1 score        : %f\n"%str(self.f1score) 
                cls_model_info += "| Model robustness score: %f\n"%str(self.rf_score)
            except:
                pass 
            cls_model_info += "| Training batch size   : %d\n"%self.cls_batch_size
            cls_model_info += "| Trained on            : %s\n"%self.processor_type
            cls_model_info += "| \n"
            cls_model_info += "|===========================================================|\n"

            summary_fname = self.class_model_id  + "_" + str(self.cls_input_dim) + "pxl_" + self.cls_tf_model_type + "_model_summary.txt"

            if write_to_file:
                with open(self.cls_model_path  +"/"+ summary_fname, 'w') as model_summary:
                    model_summary.write(cls_model_info)
                    self.logger("\n[%s] | Classifier model summary written successfully..."%time_stamp())
                
                with open(self.cls_model_path +"/"+ self.class_model_id +".model" , 'w') as model_id:   # write model ID file
                	model_id.write(",".join([each_class for each_class in self.cls_labels]))               

                self.logger("\n[%s] | Classifier model summary written successfully..."%time_stamp())


            else:
                self.logger( "\n" + cls_model_info, color = 4)                                       # show money 

        except:
            self.logger("\n[%s] | WARNING! Failed to write or display Classifier model summary due to unknown error..."%time_stamp(), color = 5)
            pass


    def make_cls_dataset(self):
        self.update_vars()                      # updatet the buttons settinsg from UI
        if self.build_entire_dataset:           # if make entire dataset
            self.make_entire_cls_dataset()
        else:
            self.make_single_class_dataset()    # if make single class dataset


    def make_single_class_dataset(self):
        if self.qm.question(self,'CRISP', "Construct classifier dataset (constructs only one class per operation)?" , self.qm.Yes | self.qm.No) == self.qm.No:
            return

        self.run_mode = "classifier_dataset"
        self.logger("[%s] | Preforming dataset construction for classifier class : %s"%(time_stamp(),self.cls_dataset_classID), color = 3)
        self.logger("\n[%s] | WARNING! All previous files in target folder (if exists) WILL BE DELETED!"%time_stamp(), color = 5) if self.clear_existing_files else None

        org_list    = get_source_images(self.cls_dataset_src_path)             # get the jpg, jpeg or png images                                  
        total_files = len(org_list)                                            # get total number of files (no sub dirs)

        self.logger("[%s] | Total source contour files found  : %d"%(time_stamp(),total_files) , color =4 )
        self.logger("[%s] | Training : Validation ratio       : [ %f : %f ]"%(time_stamp(),self.cls_train_ratio, (1-self.cls_train_ratio)), color =4  )

        train_list =  random.sample(org_list, int( self.cls_train_ratio * total_files))
        test_list  =  unused_list (org_list, train_list )                     # return the files and pout in testing set that are not in the train list


        if self.clear_existing_files:
            try:
                shutil.rmtree(self.cls_dataset_dst_path + self.cls_dataset_id + "/Train/" + self.cls_dataset_classID) 
            except:
                pass
            try:
                shutil.rmtree(self.cls_dataset_dst_path + self.cls_dataset_id + "/Test/" + self.cls_dataset_classID)
            except:
                pass

        os.makedirs(self.cls_dataset_dst_path + self.cls_dataset_id + "/Train/" + self.cls_dataset_classID, exist_ok=True)           # cosntruct the folders for dataset
        os.makedirs(self.cls_dataset_dst_path + self.cls_dataset_id + "/Test/"  + self.cls_dataset_classID, exist_ok=True)           # construct the fodlers for class



        tqdm_info = "[%s] | Processing: Training contour dataset "%time_stamp()                     # copy train list
        for file in tqdm(train_list, desc=tqdm_info, ncols =100):
            shutil.copy(file, self.cls_dataset_dst_path + self.cls_dataset_id  + "/Train/" + self.cls_dataset_classID  + "/" + os.path.basename(file))

        tqdm_info = "[%s] | Processing: Validation contour dataset"%time_stamp()                   # copy test list
        for file in tqdm(test_list, desc=tqdm_info, ncols =100):
            shutil.copy(file, self.cls_dataset_dst_path + self.cls_dataset_id + "/Test/" + self.cls_dataset_classID  + "/" + os.path.basename(file))

        self.logger("[%s] | Dataset constructed successfully with for %s --> Train --> %s"%(time_stamp(), self.cls_dataset_id, self.cls_dataset_id ) , color = 4 )         
        self.logger("[%s] | Dataset constructed successfully with for %s --> Test  --> %s"%(time_stamp(), self.cls_dataset_id, self.cls_dataset_id ) , color = 4)  


    #=========================================== Making dataset for classifier

    def make_entire_cls_dataset(self):
        if self.qm.question(self,'CRISP', "Construct entire classifier dataset (all classes in base folder are constructed at once)?" , self.qm.Yes | self.qm.No) == self.qm.No:
            return
        self.run_mode = "classifier_dataset"

        self.logger("\n[%s] | WARNING! All previous files in target folder (if exists) WILL BE DELETED!"%time_stamp(), color = 5) if self.clear_existing_files else None

        classes_path = glob.glob(self.cls_base_datasrc_path +"/*") # get list of all directories

        self.logger("\n[%s] | Preforming dataset construction for classifier"%time_stamp(), color =3)
        self.logger("[%s] | Total number of classes found     : %d"%(time_stamp(),len(classes_path)) , color =4 )


        for each_class in classes_path:

            self.cls_dataset_src_path = each_class                                 # make subsituite as for each class
            self.cls_dataset_classID  = os.path.basename(each_class)               # in this case, the base class name will be the folder name
            self.logger("\n[%s] | Processing dataset for class      : %s"%(time_stamp(), self.cls_dataset_classID ), color = 2)


            org_list    = get_source_images(self.cls_dataset_src_path)             # get the jpg, jpeg or png images                                  
            total_files = len(org_list)                                            # get total number of files (no sub dirs)

            self.logger("[%s] | Total source contour files found  : %d"%(time_stamp(),total_files) , color =4 )
            self.logger("[%s] | Training : Validation ratio       : [ %f : %f ]"%(time_stamp(),self.cls_train_ratio, (1-self.cls_train_ratio)), color =4  )

            random.seed(self.database_rseed_val.value())                           # random seed value
            train_list =  random.sample(org_list, int( self.cls_train_ratio * total_files))
            test_list  =  unused_list (org_list, train_list )                     # return the files and pout in testing set that are not in the train list


            if self.clear_existing_files:
                try:
                    shutil.rmtree(self.cls_dataset_dst_path + self.cls_dataset_id + "/Train/" + self.cls_dataset_classID) 
                except:
                    pass
                try:
                    shutil.rmtree(self.cls_dataset_dst_path + self.cls_dataset_id + "/Test/" + self.cls_dataset_classID)
                except:
                    pass


            os.makedirs(self.cls_dataset_dst_path + self.cls_dataset_id + "/Train/" + self.cls_dataset_classID, exist_ok=True)           # cosntruct the folders for dataset
            os.makedirs(self.cls_dataset_dst_path + self.cls_dataset_id + "/Test/"  + self.cls_dataset_classID, exist_ok=True)           # construct the fodlers for class

            # Iterate through each files and copy accordingly
            tqdm_info = "[%s] | Processing: Training contour dataset "%time_stamp()                     # copy train list
            for file in tqdm(train_list, desc=tqdm_info, ncols =100):
                shutil.copy(file, self.cls_dataset_dst_path + self.cls_dataset_id  + "/Train/" + self.cls_dataset_classID  + "/" + os.path.basename(file))

            tqdm_info = "[%s] | Processing: Validation contour dataset"%time_stamp()                   # copy test list
            for file in tqdm(test_list, desc=tqdm_info, ncols =100):
                shutil.copy(file, self.cls_dataset_dst_path + self.cls_dataset_id  + "/Test/" + self.cls_dataset_classID  + "/" + os.path.basename(file))

            self.logger("[%s] | Dataset constructed successfully with for %s --> Train --> %s"%(time_stamp(), self.cls_dataset_id, self.cls_dataset_classID ) , color = 4 )         
            self.logger("[%s] | Dataset constructed successfully with for %s --> Test  --> %s"%(time_stamp(), self.cls_dataset_id, self.cls_dataset_classID ) , color = 4)  


        # Uses pre-set percentages of simpels from trainin gset to create Inference samples (% of trainin gset) here
        if self.flag_use_inference_sampling.isChecked():
            try:
                shutil.rmtree(self.cls_dataset_dst_path + self.cls_dataset_id + "/Inference_samples")                
            except:
                pass

            self.build_inference_samples( self.cls_dataset_dst_path + self.cls_dataset_id + "/Train/" ,          # source folder
                                          self.cls_dataset_dst_path + self.cls_dataset_id  ,                     # destination folder where Inference_sampels folder will be created (i.e /Train/)
                                          self.perc_sample_inference_val.value(),           # percentae of sampels within Tarining classes
                                          self.database_rseed_val.value()  )                # random seed
            
            self.logger("[%s] | Dataset constructed successfully for %s --> Inference Sampling"%(time_stamp(), self.cls_dataset_id) , color = 4)        
        
        self.logger("\n[%s] | Dataset construction completed..."%time_stamp(), color = 4)  


    def build_inference_samples(self, source_folder, destination_folder, percentage=10, random_seed=None):
        # Set random seed for reproducibility
        self.logger("[%s] | Building Inference samples from %d%% of Training data"%(time_stamp(), self.perc_sample_inference_val.value() ) , color = 4 )
        destination_folder = os.path.join(destination_folder, "Inference_samples")
        os.makedirs(destination_folder, exist_ok=True)  # xontruct inference samples

        for folder in os.listdir(source_folder):
            folder = os.path.join(source_folder,folder)           
            jpg_files = [os.path.join(folder,file) for file in os.listdir(folder) if (file.lower().endswith('.jpg') or file.lower().endswith('.png'))]            # Get a list of all full pathnmaneJPEG files in the source folder        
      
            num_files_to_move      = int(len(jpg_files) * (percentage / 100))                             # Calculate the number of files to move based on the specified percentage        
            random.seed(random_seed)
            files_to_move          = random.sample(jpg_files, num_files_to_move)                          # Randomly select files to move

            inference_folder_class = os.path.join(destination_folder,os.path.basename(folder))            # full folder pathname for destination folder  
            os.makedirs(inference_folder_class, exist_ok=True)                                            # contruct inference samples              
            
            tqdm_info = "[%s] | Processing: Creating Inference samples "%time_stamp()                     # copy train list
            for file in tqdm(files_to_move,desc=tqdm_info, ncols =100):                                   # Move selected files to the destination folder & Iterate through each files and copy accordingly
                try:    
                    shutil.move(file, inference_folder_class)
                    #print(f"Moved: {file} to {inference_folder_class}")
                except:
                    prinf(f'Failed moving:{file} to {inference_folder_class}')


