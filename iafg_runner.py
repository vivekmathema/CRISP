#!/usr/bin/python
# CRISP II: Leveraging Deep Learning for Multiclass Phenotyping in GC×GC-TOFMS of End-Stage Kidney Disease Patients
# Mathema et al 2024 : Siriraj Metabolomics & Phenomics Center (SiMPC), Mahidol University, Bangkok, Thailand
# Software code under review VERSION
# encoding: utf-8
# A part of source code file for the CRISPII. (Under Review)

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
import numpy as np
import glob
from tqdm import tqdm
from itertools import cycle
from utils_tools import *

def get_source_images(scan_path):
    src_filelist =[]                                                                  # iniciate the list of files
    src_types = (scan_path + '/*.jpg',                                                # inciate glob list for multiple possible file type (make in module later)
                 scan_path + '/*.jpeg',
                 scan_path + '/*.png'  )    
    for files in src_types:
        src_filelist.extend(glob.glob(files))

    return src_filelist


def img_to_vid(fpath ="./source/", fout ="./cyclic_vid.mp4", fps = "auto", duration = "auto" , 
              coding ="mp4v",  sharpen_flag = False , new_dim = None,
              agu_dilate   = False,
              agu_erode    = False,
              agu_noised   = False,
              agu_denoised = False):                                                                 # Function to give resperentative imge of each class

	print('Reading source images....', end ="")	
	image_data   = [cv2.imread(img) for img in tqdm(get_source_images(fpath))]

	#===============================                                                                 # Append agumented images if set by user
	tmp_img_data = []
	for img in tqdm(image_data ,desc = "\n[%s] Processing possible agumentation(s) "%time_stamp(), ncols =100):
		tmp_img_data.append(img)                                                                     # append the image
		if agu_dilate:
			tmp_img_data.append(img_dilate(img,(3,3), 1) ) 
		if agu_erode:
			tmp_img_data.append(img_erode(img, (3,3), 1) ) 
		if agu_noised:
			tmp_img_data.append(agument_noise(img, "gaussian", 5, 5) ) 
		if agu_denoised:
			tmp_img_data.append(rnd_denoise(img, kernel_range = 20 ) )

	#==================================
	if len(tmp_img_data) > len(image_data):                                                          # check if image agumentation has been done
		image_data = tmp_img_data                                                                    # replace 	
	del tmp_img_data                                                                                 # free up memory


	print("Final size of the image_data:", len(image_data))

	if new_dim != None:                                                                              # resize all imegs to the given dimension if new_dim is supplied
		print("\n[%s] | Resizing images before AFRC processing to dimensions (w,h): (%d, %d) pixels\n"%(time_stamp(), new_dim[0],new_dim[1]) )   # new_dim = new_shape ==(w,h)
		image_data   = [cv2.resize(img, new_dim) for img in tqdm(image_data)]		

	if sharpen_flag:
		print("[%s] | Applying AFRC image sharpening"%time_stamp())
		image_data   = [rnd_sharpen(img) for img in tqdm(image_data)]

	img_array    = []

	if fps == "auto":
		fps = 15

	if duration == "auto":
		duration = int( len (image_data)/fps * 25 )

	count        = 0
	total_frames = fps * duration
	pool         = cycle (image_data)
	print("[%s] | Total video duration (in seconds): %0.2f"%(time_stamp(), total_frames/fps) )

	while count < total_frames:
		img_array.append(next(pool))
		count +=1

	height, width, layers = img_array[0].shape                              # take height and width of video from first frame image of array
	size = (width,height)
	# open output stream for writing write image with  fps
	out = cv2.VideoWriter(fout,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)  #mp4v for .MP$ and AVI for .avi

	print('[%s] | Creating cyclic frame rotation video from images....'%time_stamp())
	for i in tqdm(range(len(img_array))):
		out.write(img_array[i])
	out.release()

	return fout


def scaled_resize(img, scale = 85): # scale in percetage                    # this is nly for preview 
	scale_percent = scale
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	# resize image
	return cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 


# loop over images and estimate background
def avg_weight(fpath = "./source/source_video.mp4", win_title ="CLASS-X", show_preview =True):

	print("Source path:", fpath)

	c = cv2.VideoCapture(fpath) 
	_,f = c.read()

	avg2 = np.float32(f)
	print("[%s]Processing : %s"%(time_stamp(),win_title))
	for x in tqdm(f):
		_,f = c.read()
		if f is  None:
			cv2.destroyAllWindows()
			print("\n[%s] | Optimization completed earlier then routined..."%time_stamp())
			return res2                          # returns the best possible figure
		else:
			cv2.accumulateWeighted(f,avg2,0.01)
			res2 = cv2.convertScaleAbs(avg2)
			
		if show_preview:
			f       = cv2.copyMakeBorder(f,5,5,5,5,cv2.BORDER_CONSTANT,value= [0,0,0])
			tmp_res = res2
			tmp_res = cv2.copyMakeBorder(tmp_res,5,5,5,5,cv2.BORDER_CONSTANT,value= [0,0,0])
			cv2.imshow("Processing AFRC: %s | TOP: Sample images| BOTTOM: Single AFRC Image"%win_title, cv2.vconcat( ( scaled_resize(f), scaled_resize(tmp_res))   )  )

		k = cv2.waitKey(1) & 0xff
		if k == ord('q') or k == 27:             # press q or escape\
			cv2.destroyAllWindows()
			print("\nWarning! returning pre-maturely termianted image by user. The image may not have reached its optimization...")
			break
	cv2.destroyAllWindows()
	c.release()                                                   # release the 
	return  res2

