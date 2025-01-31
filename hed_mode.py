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

import cv2
import os
from   tqdm import tqdm
import glob
import random
import seaborn as sns
import matplotlib.pyplot as plt
import random

from utils_tools import *


class CropLayer(object):
	def __init__(self, params, blobs):
		# initialize our starting and ending (x, y)-coordinates of the crop
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0

	def getMemoryShapes(self, inputs):
		# the crop layer will receive two inputs -- we need to crop
		# the first input blob to match the shape of the second one,
		# keeping the batch size and number of channels
		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])

		# compute the starting and ending crop coordinates
		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H

		# return the shape of the volume (we'll perform the actual # crop during the forward pass		
		return [[batchSize, numChannels, H, W]]

	def forward(self, inputs):
		# use the derived (x, y)-coordinates to perform the crop
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]


#============================================================ Gradient converted image array 
def get_gradient_images(image_array, grad_kernel = (3,3) , grad_rnd_fac = 0) :
    grad_img_array =[]
    for each_img in image_array:
    	if not grad_rnd_fac ==0:
    		grad_kernel =  (  max(1,int(grad_rnd_fac* random.random()  )  ),  
    			              max(1,int(grad_rnd_fac* random.random()  )  )   )

    	grad_img_array.append( cv2.morphologyEx(each_img, cv2.MORPH_GRADIENT, grad_kernel)  )
    return grad_img_array

#=================================================================Get Gradient image


def convert_contour_to_heatmap_with_threshold(contour_image, threshold=128):
    # Apply colormap to create a heatmap
    contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(contour_image, cv2.COLORMAP_JET)

    # Apply threshold to the heatmap
    _, thresholded_heatmap = cv2.threshold(contour_image, threshold, 255, cv2.THRESH_BINARY)

    # Set pixels in the thresholded heatmap to black in the original heatmap
    heatmap[thresholded_heatmap == 0] = [0, 0, 0]

    return heatmap


#============================================================ Gradient converted image array 
def get_threshold_images(image_array, threshold = 128) :
    thsi_img_array =[]
    for each_img in tqdm(image_array, desc="Processing fixed gradient thresholding of images"):
    	img = convert_contour_to_heatmap_with_threshold (each_img, threshold)
    	thsi_img_array.append(img)
    return thsi_img_array

#=================================================================Get Gradient image

'''
Adaptive Gradient Thresholding
The threshold_value parameter is used for setting a threshold to binarize the gradient magnitude image obtained after Sobel edge detection. 
Even though we are using adaptive thresholding to minimize noise, we still need to set a threshold value to distinguish between the edges (where the gradient magnitude is high)
 and the background (where the gradient magnitude is low).In summary, there is no one-size-fits-all answer for choosing the best order value. It often involves a combination 
 ksize_x and ksize_y control the scale of the Sobel operation in the horizontal and vertical directions respectively. Adjusting these parameters can affect the sensitivity of edge detection and the level of detail captured in the gradient images.
'''

def adaptive_gradient_threshold(image_array, threshold_value=128, output_mode='RGB', ksize_x=3, ksize_y=3, sort_option='sort', rndseed = 42, show_store_plot_fname = None):
    # Set random seed for reproducibility
    random.seed(rndseed)
    thsi_img_array =[]
    trendline_data = []
    
    # Progress bar setup
    for img in tqdm(image_array, desc="Processing Adaptive gradient thresholding:"):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # Convert to grayscale
        
        # Compute the gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize_x)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize_y)        
        
        magnitude = cv2.magnitude(grad_x, grad_y)                                             # Compute the magnitude of the gradients        
        
        _, binary_output = cv2.threshold(magnitude, threshold_value, 255, cv2.THRESH_BINARY)  # Apply the threshold
        
        if output_mode.lower() == 'rgb':
            # Create a mask from the binary output
            mask = binary_output.astype(np.uint8)
            # Apply the mask to the original image (preserving color)
            output_img = cv2.bitwise_and(img, img, mask=mask)
        elif output_mode.lower() == 'grayscale':
            # Use the binary output as the final image
            output_img = binary_output.astype(np.uint8)        
        
        thsi_img_array.append(output_img)                                                        # Save the processed image to image_array
        
        # Collect data for trendline
        avg_magnitude = np.mean(magnitude)
        trendline_data.append(avg_magnitude)
    
    
    if sort_option.lower() == 'sort':                                                            # Sort, unsort, or shuffle the trendline data
        sorted_trendline_data = sorted(trendline_data)
    elif sort_option.lower() == 'unsort':
        sorted_trendline_data = trendline_data
    elif sort_option.lower() == 'shuffle':
        random.shuffle(trendline_data)
        sorted_trendline_data = trendline_data

    if show_store_plot_fname != None:
    	# Plot trendline
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=np.arange(len(sorted_trendline_data)), y=sorted_trendline_data, color='blue')
        sns.regplot(x=np.arange(len(sorted_trendline_data)), y=sorted_trendline_data, order=5, scatter=False, color='red')
        plt.title(f'sample destribution magnitude', fontsize=22)
        plt.xlabel(f'sample index ({sort_option})', fontsize=20)
        plt.ylabel('average gradient magnitude', fontsize=20)
        plt.tight_layout()
        try:
            plt.savefig(show_store_plot_fname)
        except:
            print("Warning! Cannot save the tendline graph...")
        plt.show()
        plt.close()

    return thsi_img_array

