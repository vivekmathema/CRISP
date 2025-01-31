#!/usr/bin/python
# CRISP II: Leveraging Deep Learning for Multigroup Classification in GC×GC-TOFMS of End-Stage Kidney Disease Patients
# Mathema et al 2025 : Siriraj Metabolomics & Phenomics Center (SiMPC), Mahidol University, Bangkok, Thailand
# Software code under review
# encoding: utf-8
# Main source code file for the CRISP. Under Review

'''
     ░▒▓██████▓▒░░▒▓███████▓▒░░▒▒▓█▓▒░░▒▓███████▓▒░▒▓███████▓▒░  ░▒▓█▓▒░ ▒▓█▓▒░
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒  ▒▓█▓▒░▒▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒  ░▒▓█▓▒░ ▒▓█▓▒░
    ░▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒░▒▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒  ░▒▓█▓▒░ ▒▓█▓▒░ 
    ░▒▓█▓▒░      ░▒▓███████▓▒░░▒▒▓█▓▒░░▒▓██████▓▒░░▒▓███████▓▒░  ░▒▓█▓▒░ ▒▓█▓▒░
    ░▒▓█▓▒░      ░▒▓█▓▒░░▒▒▓█▓░▒▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░        ░▒▓█▓▒░ ▒▓█▓▒░ 
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▒▓█▓░▒▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░        ░▒▓█▓▒░ ▒▓█▓▒░
     ░▒▓██████▓▒░░▒▓█▓▒░░▒▒▓█▓░▒▒▓█▓▒░▒▓███████▓▒░░▒▓█▓▒░        ░▒▓█▓▒░ ▒▓█▓▒░
------------------------------------------------------------------------------------------------------------------------------------------------------------------
__author__      = "Mathema VB, Duangkumpha K, Wanichthanarak K, Jariyasopit N, Dhaka E, Sathirapongsasuti N, Kitiyakara C,Sirivatanauksorn Y, Sakda Khoomrung S*  |
__institute__   = "Metabolomics and Systems Biology,Department of Biochemistry,Faculty of Medicine Siriraj Hospital, Mahidol University, Bangkok 10700, Thailand" |
__license__     = "GNU/GPL"                                                                                                                                       |
__version__     = "0.01"                                                                                                                                          |
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

'''

from skimage.metrics import structural_similarity 
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error
import imutils
import cv2
import math
import seaborn as sns
import matplotlib.pyplot as plt 
import keras
import time
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
import numpy as np
import tensorflow as tf
#==========
from scipy.spatial import distance
import scipy.misc
#==========
from keras.applications.inception_v3 import InceptionV3
from fid_module import compute_fid
#==========
from siamese_net import siamese_model, siamese_score


#///////////////////////////////////////////////////////
def show_SSIMGraph(SSIM_graph, winsize, win_shift, show_ssim_graph=False, store_graph=None):
    if len(SSIM_graph) > 0:
        plt.ioff()  # Turn off interactive plotting
        sns.set(style="whitegrid")  # Set the style for seaborn plots
        fig, ax = plt.subplots()

        x_vals = [x[1] for x in SSIM_graph]
        y_vals = [y[0] for y in SSIM_graph]

        sns.lineplot(x=x_vals, y=y_vals, ax=ax)
        sns.scatterplot(x=x_vals, y=y_vals, ax=ax)

        ymax, ymin = max(y_vals), min(y_vals)
        xpos_max, xpos_min = y_vals.index(ymax), y_vals.index(ymin)
        xmax, xmin = x_vals[xpos_max], x_vals[xpos_min]

        ax.annotate(' max', xy=(xmax, ymax), xytext=(xmax, ymax),
                    arrowprops=dict(facecolor='black', shrink=0.02), fontsize=16)
        ax.annotate(' min', xy=(xmin, ymin), xytext=(xmin, ymin),
                    arrowprops=dict(facecolor='black', shrink=0.02), fontsize=16)

        ax.set_xlabel(f'scan: R1 lengthwise | Win size: {winsize} | Win. shift: {win_shift}', fontsize=18)
        ax.set_ylabel('similarity scores @ scan win. midpoint', fontsize=18)
        ax.set_title("sliding window for similarity scores", fontsize=18)

        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to ensure everything fits
        
        if store_graph:
            plt.savefig(store_graph)
            print("[INFO] | ROIs graph saved: %s"%store_graph)
        plt.close(fig)

        if show_ssim_graph == True:                                                        # open show image just 3 seconds
            plt.show() 
        else:
            plt.ion()
            plt.show()
            time.sleep(3)                                                                # show for 3 secodns and off
            plt.close('all')
            plt.ioff()

        plt.ion()                                                                        # turn back the plot in line (plt.ion()) 
        return                                                                           # return the graph


def vgg16_ssim(contour_A,contour_B):                                                                 # returns the image similarity based on VGG16 network
	global imageA, imageB, vgg16_scan, basemodel

	# SUBMODULE FUNCTIONS
	def get_feature_vector(img):
		img1 = cv2.resize(img, (224, 224))
		feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
		return feature_vector

	def read_image(img):                                                                                       
		return cv2.imread(img)

	def calculate_similarity(vector1, vector2):                                                       #Define another function for calculating cosine similarity.                      
		return 1 - distance.cosine(vector1, vector2)
	#===================

	fsegA = get_feature_vector(contour_A)
	fsegB = get_feature_vector(contour_B)

	SSIM_score =  calculate_similarity(fsegA, fsegB)

	return SSIM_score                                                                            #returns score as flot range(0-1)
                                                                    
def stacked_img(img, stacking_array):                                                            # function to horizontall stack an image array
	stack_img =[]                                                                                # based on cv2 image and position as arrays
	for pos in stacking_array:
		start,end = pos
		stack_img.append(img[ :, start:end ])
	return cv2.hconcat(stack_img)


def hamming_distance(imgA, imgB):                                                                # 1-score is doen to make the value of similar to 100%
	score =scipy.spatial.distance.hamming(imgA.flatten(), imgB.flatten())
	return (1-score)


# function to select maximum image difference region of interest of traning portion
def ssim_diff(contour_A,contour_B, winsize= 24, scan_step = 1, win_shift = 10,   ssim_thresthold = 0.94, show_scan = False , SSIM_mode ="SSIM"):
	global imageA, imageB, vgg16_scan, basemodel , fid_model, siam_model
	# compute the Structural Similarity Index (SSIM) between the two  images, ensuring that the difference image is returned

	#==================================================================Image simlarity calculation block (also contains experimental score matrix for their standard use)

	if SSIM_mode == "SSIM":                                                                      # https://scikit-image.org/docs/dev/api/skimage.metrics.html, returns (mean strictural sim.) 
		(SSIM_score, diff) = structural_similarity(contour_A, contour_B, full=True)              # Basic, calculate the structureal simiatirty uptop sixth precission, SSIM_score is float ( 0-1 )		
		diff = (diff * 255).astype("uint8")                                                      # for SSIM,PNSR mode only.  Make it for 0-255 grayscale image (returns gryscale image)
		thresh  = cv2.threshold(diff, 0, 255, cv2.THRESH_TRUNC | cv2.THRESH_OTSU)[1]             # for simple mode only. Shows image difference threshtold choose between cv2.THRESH_BINARY cv2.THRESH_BINARY_INV , 'BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV'

	if SSIM_mode == "PNSR":                                                                      # https://scikit-image.org/docs/dev/api/skimage.metrics.html, returns (mean strictural sim.)      
		SSIM_score  = peak_signal_noise_ratio(contour_B, contour_A)/100                          # returns (float) peak signal to noise ratio (PSNR) for an image (make it between 1-0)		
		print( " --> PNSR :", SSIM_score )		                                                 #  #cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #identify contours (retursn list of each countour locations) code:

	if SSIM_mode == "HAMMING":                                                                   # https://scikit-image.org/docs/dev/api/skimage.metrics.html, returns (mean strictural sim.)      
		SSIM_score  = hamming_distance(contour_A, contour_B)                                     # returns (float) peak signal to noise ratio (PSNR) for an image (make it between 1-0)		
		print( " --> HAMMING :", SSIM_score )	

	if SSIM_mode == "VGG16":                                                                     # implement state-of-art VGG16 based matrix CNN image differences
		SSIM_score = vgg16_ssim(contour_A,contour_B)                                             # returns score

	if SSIM_mode == "FID":                                                                       # implement state-of-art InceptionV3 based FID model for image differences
		SSIM_score = compute_fid(fid_model, contour_A, contour_B)                                # returns score (it may not ber between 0-1 so the thresthold is deactivated)

	if SSIM_mode == "SIAMESE":                                                                   # implement state-of-art InceptionV3 based FID model for image differences
		SSIM_score = siamese_score(siam_model, contour_A, contour_B)                             # returns score (it may not ber between 0-1 so the thresthold is deactivated)


	#====================================================================Image display block
	win_start = (win_shift * scan_step)                                                          #  start pos of window
	win_end   = (win_shift * scan_step) + winsize                                                # end pos of window   (please note there will be one pixel common betwen each consecutive window scan)

	tmpA, tmpB =imageA.copy() , imageB.copy()                                                    # make teh copy of image A which we are going to be shifting frames
                                                                                                 # make teh copy of image A which we are going to be shifting frames

	# put border and stack imaegs worwise                                                         https://stackoverflow.com/questions/42420470/opencv-subplots-images-with-titles-and-space-around-borders
	if  show_scan:                                                                               # Show the sliding windows region being processed and the output images
		contA = cv2.copyMakeBorder(contour_A,10,10,10,10,cv2.BORDER_CONSTANT,value= [0,0,0])     # vorder colour value = bloack i.ed [0,0,0]  
		contB = cv2.copyMakeBorder(contour_B,10,10,10,10,cv2.BORDER_CONSTANT,value= [0,0,0])
		cv2.imshow("Class-A (TOP), Class-B (Bottom)", cv2.vconcat((contA, contB)))
		cv2.imshow("Difference", diff)                      if SSIM_mode == "SSIM" else None     # show thresthold only in case of SSIM simple                        
		#cv2.imshow("Differenciation Thresthold", thresh) if SSIM_mode == "SSIM" else None       
		#cv2.imshow("Scannig:  Image Class A", tmpA)
		#cv2.imshow("Scanning: Image Class B", tmpB)      
		cv2.waitKey(1)                                                                           # wait for 100ms and pass on 

	#======================================================================================

	if SSIM_score <= ssim_thresthold and SSIM_mode != "FID":
		return SSIM_score                                                                        # return  scan_step (segment) , scan window(start-end), SSIM_score, No of contours, window size
	elif  SSIM_mode == "FID":                                                                    # for FID mode tehre is No thresthold. only scores will be arranged
		return SSIM_score

	else:
		return None                                                                              # return None if below thresthold
 

#/////////////////////////////////////////////////////// Image input and parameetrs
def scan_similarity(imgA, imgB, ssim_thresthold = "auto", winsize ="auto", win_shift = "auto" , 
	                win_segment = "auto", show_scan = True, show_ssim_graph = True , SSIM_mode = "SSIM", avoid_overlap = True , store_graph = None):
	global imageA, imageB , vgg16_scan, basemodel , fid_model , SSIM_graph, siam_model

	# convert the RGB images to grayscale
	imageA, imageB = imgA, imgB                                                                        # load the two original datasource input images
	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY) if SSIM_mode == ("SSIM" or "PNSR") else imgA      # vgg16 required 3 channel image, while simple mode required only grayscale
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY) if SSIM_mode == ("SSIM" or "PNSR") else imgB      # vgg16 requires 3 channel image, while simple mode required only grayscale

	height, width  = grayA.shape[0], grayA.shape[1]
	ssim_scores    = []                                                                                # holds the scores
	SSIM_graph     = []

	if win_segment =="auto":
		win_segment = round(width /250)
                                                                                           # current total semnets
	if winsize == "auto":
		winsize =math.floor(width/10)                                                      # winsize : Automaticic the window size that need to be scanned making highest number of desired segemnts
                                                                                           # this value seesm to neglect the first 300-500s of reading that have noicse 
	if win_shift == "auto":
		win_shift = winsize - math.floor(winsize/50)                                       # set auto value for winsihft ,# width/50 is used for making th width slightly smaller to avoid segemntation error while looping windows 

	if avoid_overlap == True:
		win_shift = winsize                                                                # Overwrites the winsize (even in auto mode) vlaue and sets the vlaue equal to winow size to avoid overlapping


	if ssim_thresthold == "auto":
		ssim_thresthold  = 0.95                                                            # set auto value for SSIM thresthold
	                                                        

	print("\n===================================\n")
	if SSIM_mode == "VGG16":
		print("-->VGG16 mode selected. Loading VGG16 CNN filter...")
		fname ="./base_weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5"                                 # downlad ="imagenet"
		vgg16_scan = keras.applications.VGG16(weights=fname, include_top=True, pooling="max", input_shape=(224, 224, 3))	
		basemodel = Model(inputs=vgg16_scan.input, outputs=vgg16_scan.get_layer("fc2").output)               #We don’t need all layers of the model. Extract vector from layer “fc2” as extracted feature of image.

	if SSIM_mode == "FID":
		print("-->FID mode selected (very slow processing.... Loading FID filter...")                       
		fid_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))                  #load the base model (fid_model) for FID based on inceptionV3 and share with the scoring module 

	if SSIM_mode == "SIAMESE":
		print("-->SIAMESE NET mode selected. Loading SIAMESE Network weights..")                       
		siam_model = siamese_model()                                                                        #load the base model (fid_model) for FID based on inceptionV3 and share with the scoring module 

		
	# show infromations & start scanning

	print("-->Source image dim.(h,w): (%d , %d) "%(height,width) )
	print("-->Scan windows size     :", winsize)
	print("-->Scan segments         :", win_segment)
	print("-->Scan window shift     :", win_shift)

	scan_win  = 0                                                         # initalixe scan_win position                                                                      
	scan_step = 0                                                         # initalize scan_step       
	while ( (scan_win + winsize) <= width ):                              # width-width/100 is used instead of width to make the segment count same as required       
		contour_A = grayA[: , scan_win: scan_win  + winsize ]
		contour_B = grayB[: , scan_win: scan_win  + winsize ]
		scan_result = ssim_diff(contour_A,contour_B, winsize , scan_step = scan_step, 
			                                                   win_shift = win_shift, 
			                                                   ssim_thresthold =ssim_thresthold, 
			                                                   show_scan = show_scan,
			                                                   SSIM_mode = SSIM_mode)

		scan_status = round( (scan_win + winsize)/width * 100 )             # scan_results gets (scan_step, (win_start, win_end), SSIM_score, winsize)
		print ("\r-->Scan pos : ( %d - %d ) | Scanning ROI: %d%%"%(scan_win, 
		                                                    scan_win + winsize, 
		                                                    scan_status ),  end ="" )    # Print is same line  

		if (scan_result !=None):                                                                       # append the results
			SSIM_graph.append( (scan_result, (scan_win + winsize/2) ))                                 # if the resturn type is not NuLL, add the ssim and the scan window mid position
			ssim_scores.append((scan_step, (scan_win, scan_win + winsize), scan_result, winsize))      # win_start is scan_win, win_end-pos is scan_win + winsize
			print("  --> ",ssim_scores[-1:]) 

		scan_win  += win_shift                                                                         # window shift size 
		scan_step += 1

                                                                                         
	print("\n===================================\n")
	#print("Original list:\n", sim_scores )
	SSIM_sorted = sorted(ssim_scores, key=lambda k: k[2])                                        # sort based on increasing SSIM vlaue , Sorting key =SSIM list: (scan_step, (win_start, win_end), SSIM_score, len(cnts), winsize)
	
	score_plt = show_SSIMGraph(SSIM_graph, winsize, win_shift, show_ssim_graph , store_graph ) 	# Show graph  fi needed using store_graph as filepath

	cv2.destroyAllWindows()                                                                     # close all scanning windows

	if SSIM_sorted != []:
		return  SSIM_sorted 
	else:
		 return None 




