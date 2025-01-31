#!/usr/bin/python
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
#============================================

import cv2
# these must be global variable  
# the roi_image variable is global as well but is shared  here between 2 modules
refPt = []
cropping = False
win_id = "ROI | Press [c]: ROI selection | [r]: Reset | [q/Esc]: Quit module"         # windows id with information

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global roi_image, refPt, cropping
	# if the left mouse button was clicked, record the starting  (x, y) coordinates and indicate that cropping is being performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that the cropping operation is finished
		refPt.append((x, y))
		cropping = False
		# draw a rectangle around the region of interest
		cv2.rectangle(roi_image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow(win_id, roi_image)

def get_rois_cord(img):                                                   # input in cv2.image, the finction returns co-ordinates of the rois
	global roi_image, refPt, cropping
	roi_image =img
	clone = roi_image.copy()

	cv2.namedWindow(win_id)
	cv2.setMouseCallback(win_id, click_and_crop)
	# keep looping until the 'q' key is pressed
	while True:
		# display the image and wait for a keypress
		cv2.imshow(win_id, roi_image)
		key = cv2.waitKey(1) & 0xFF
		# if the 'r' key is pressed, reset the cropping region
		if key == ord("r"):
			refPt = []
			cropping = False
			roi_image = clone.copy()
		# if the 'c' key is pressed, break from the loop
		elif key == ord("c"):
			if refPt ==[]:
				print("[INFO] | No ROI roi was selected. Click LEFT Key and drag mouse to select ROI rectangle). ")
				print("[INFO] | Press [c] to select region, [r] for Resetting selection, [q] for exiting ROI selection.")
				pass
			else:
				break

		if key == ord("q") or key ==27:                                           # escape key or quit key pressed event
			print("-->Exiting without selecting roi..")
			cv2.destroyAllWindows()
			return(None)
			break
	# if there are two reference points, then crop the region of interest
	# from teh image and display it
	if len(refPt) == 2:
		roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		print("[INFO] | Selected roi co-ordinates: ", refPt[0][1],refPt[1][1], refPt[0][0],refPt[1][0] )
		print("[INFO] | Warning!!! The ROI values will be set for all images in all classes while training.")
		print("[INFO] | Must use same ROI for traninging classifier")
		#cv2.imshow("ROI 2", roi)
		#cv2.waitKey(0)
	# close all open windows
	cv2.destroyAllWindows()

	return (refPt[0][1],refPt[1][1], refPt[0][0],refPt[1][0])

