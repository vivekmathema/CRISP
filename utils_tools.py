#!/usr/bin/python
# CRISP II: Leveraging Deep Learning for Multiclass Phenotyping in GC×GC-TOFMS of End-Stage Kidney Disease Patients
# Mathema et al 2024 : Siriraj Metabolomics & Phenomics Center (SiMPC), Mahidol University, Bangkok, Thailand
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
import random
import cv2
import sys
import glob
from  PIL import Image, ImageDraw, ImageFont, ImageEnhance
from datetime import datetime
import os

def time_stamp():
    return  datetime.now().strftime("%d/%m/%Y %H:%M:%S")  

# returns the NEW LIST where LIST2 items are not in list1     
def unused_list( list1, list2):
    return [x for x in list1 if x not in list2]

def gval(value):                                                   # to circument the values which could return None causing program crash
    if value == None:
        return 0
    elif  value != None:
        return value

def check_dir(path):
    if not os.path.exists(path):                                   # try making new dir if not extists
        try:
            os.makedirs(path, exist_ok  = True)
        except:
            pass
    return

def imread(img, type='gan', img_dim = 512 , scale_fac = 1):
    x = img # imageio.imread(f)
    if type == 'gan':
        if scale_fac != 1:                    # default image be 512x512
            x = cv2.resize(x, dsize=(img_dim, img_dim), interpolation=cv2.INTER_CUBIC)
        return x.astype(np.float32) / 255 * 2 - 1                                         # return type for np float popint
    elif type == 'fid':
        x =  cv2.resize(x, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        return x.astype(np.float32)


def find_image_files(start_folder):                                                       # get images form fodlers and sub fodlers
    # Supported image file extensions
    image_extensions = ('*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.png')
    image_files = []

    if not os.path.exists(start_folder):
        print(f"ERROR! The directory [{start_folder}] does not exist.")
        return image_files

    for extension in image_extensions:        
        found_files = glob.glob(os.path.join(start_folder, '**', extension), recursive=True) # Recursively find all files matching the pattern
        for file in found_files:
            if os.path.isfile(file):
                image_files.append(file)
    
    return image_files


def get_source_images(scan_path):
    src_filelist =[]                                        # iniciate the list of files
    src_types = (scan_path + '/*.jpg',                      # inciate glob list for multiple possible file type (make in module later)
                 scan_path + '/*.jpeg',
                 scan_path + '/*.png',
                 scan_path + '/*.tiff'  )    
    for files in src_types:
        src_filelist.extend(glob.glob(files))

    return src_filelist

def zero():
    return np.random.uniform(0.0, 0.01 + random.random()/50, size = [1])

def one():
    return np.random.uniform(0.99, 1.0, size = [1])

def noise(n):
    return np.random.uniform(-1.0, 1.0, size = [n, 4096]) + random.random()/50


#===================Resize Image (based on width fit or height fit) CANNOT USE FOR THE CV2. Resize for 256 x 256 as it can make non symmetric output
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


#================================================CV2 image oeprations
def  rnd_sharpen(img , kernel_range = 9):                          # derived from https://stackoverflow.com/questions/4993082/how-to-sharpen-an-image-in-opencv
    kernel = np.array([[-1,-1,-1], 
                       [-1, kernel_range,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)                # applying the sharpening kernel to the input image & displaying it.
    #cv2.imshow('Image Sharpening', sharpened)
    return sharpened    

def rnd_denoise(img, kernel_range = 10):                     # works best for gussian noise [https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html]
    #cv2.imshow('Image Sharpening',  cv2.fastNlMeansDenoisingColored(img,None,kernel_range, kernel_range,7,21) )
    return cv2.fastNlMeansDenoisingColored(img,None,kernel_range, kernel_range,7, 21)


def rnd_flip(img, flip_type):
    # filp_type  =1  # mirror image
    # filp_type  =0  # upside down 
    # flip_type = -1 # upside down mirror
    return cv2.flip(img, flip_type)


def rnd_blurred(img, kernel_range = 3):
    return cv2.GaussianBlur(img, (kernel_range,kernel_range) , 1.0)

#=========================================================

def img_contrast(img):                                                         # agument random image contrast
    img = Image.fromarray(img)
    scale_value = min(random.random(),0.3) + 1
    return ImageEnhance.Contrast(img).enhance( 1 + scale_value )


def img_brightness(img):  
    img = Image.fromarray(img)                                                  # agument random brightness
    scale_value = min(random.random(),0.3) + 1
    return ImageEnhance.Brightness(img).enhance( 1 + scale_value )
    

def edge_enhance(img ):  #sharpen and edge enhance of images ,for randomly enhance sewt rnd_flag = True
    kernel = np.array([ [-1,-1,-1,-1,-1],
    	                [-1,2,2,2,-1],
                        [-1,2,8,2,-1],
                        [-2,2,2,2,-1],
                        [-1,-1,-1,-1,-1] ])/8.0
    return cv2.filter2D(img, -1, kernel )                                      # Always enhance and sharpen edges

#=========================================================  erode image based on kernel size and iteration                
def img_erode(img, kernel_size = (3,3), iter = 1 ):
    kernel   = np.ones(kernel_size, np.uint8)
    return  cv2.erode(img, kernel, iter)

#========================================================= dialte image based on kernelsize and iteration
def img_dilate(img, kernel_size = (3,3), iter = 1 ):
    kernel   = np.ones(kernel_size, np.uint8)
    return  cv2.dilate(img, kernel, iter)

#=========================================================

def agument_noise(img, noise_type = "gaussian" , mean = 10, var = 500):

    sigma = var ** 0.5
    sigma = sigma * 10 if noise_type == "random" else sigma
    
    if noise_type == "gaussian":
        noise = gaussian = np.random.normal(mean, sigma, (img.shape[0], img.shape[1])) 
    if noise_type == "random":
        noise =  np.random.random(size =(img.shape[0], img.shape[1]))


    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:                         # for grayscale
        noisy_image = img + noise * sigma
    else:                                           # for RGB images
        noisy_image[:, :, 0] = img[:, :, 0] + noise * sigma
        noisy_image[:, :, 1] = img[:, :, 1] + noise * sigma
        noisy_image[:, :, 2] = img[:, :, 2] + noise * sigma

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image

def set_dirpath(dirpath):
    if not dirpath.strip().endswith("/"):
        dirpath = dirpath.strip() +"/"
    return dirpath.strip()

#=============================================================                    # Crop the image to remove scale and resixe to fit the input image size of 256 x 256
def get_contour_frame_only(cv_img, height_top =17, height_bottom= 27, width_left=137, width_right=10, output_shape = (256, 256) ): # cropped the image to get only contour, values fitted to our machine output contour image here
    height,width =image.shape[:2]
    cropped_contour_img = image[height_top:height - height_bottom, width_left:width - width_right]
    cropped_contour_img = cv2.resize(cropped_contour_img, output_shape, cv2.INTER_AREA)
    return cropped_contour_img 

#========================== Image distortion module
#https://github.com/pranavsr97/Image-Processing
# Function to distort a given image, to the given co-ordinates
# an opencv2 image 'img',  x1_coord, y1_coord ... x4_coorf, y4_coord, are the given co-ordinated to which the given image is to be distorted to (supplu co ordinates clickwise or anti-clockwise)
def distorted_img(img, fac= 10 , output_dim = 256):
    shape = img.shape   # Image path and reqd new image's co-ords are read  
    height = img.shape[0]
    width  = img.shape[1] 

    #clockwise co-ordinates for distortion
    x1_coord= rnd_distort(1,fac)
    y1_coord= rnd_distort(1,fac)

    x2_coord= rnd_distort(width -1,fac)
    y2_coord= rnd_distort(1,fac)

    x3_coord= rnd_distort(width-1,fac)
    y3_coord= rnd_distort(height-1,fac)

    x4_coord= rnd_distort(1,fac)
    y4_coord= rnd_distort(height-1,fac)

    # Existing end co-ordinates of the given image
    A = [0,0]
    B = [shape[1]-1,0]
    C = [shape[1]-1,shape[0]-1]
    D = [0,shape[0]-1]

    # Storing input end co-ords in a single numpy array
    inpts=np.float32([A,B,C,D])

    # Initialising the new image's end co-ordinates
    newA=[]
    newB=[]
    newC=[]
    newD=[]

    # Defining the new image's co-ordinates
    newA.append(x1_coord)
    newA.append(y1_coord)
    newB.append(x2_coord)
    newB.append(y2_coord)
    newC.append(x3_coord)
    newC.append(y3_coord)
    newD.append(x4_coord)
    newD.append(y4_coord)

    # Storing output end co-ords in a single numpy array
    outpts = np.float32([newA,newB,newC,newD])

    # Finding minimum of all the x-coords and minimun of all the y-coords
    lowx=min(outpts[0][0],outpts[1][0],outpts[2][0],outpts[3][0])
    lowy=min(outpts[0][1],outpts[1][1],outpts[2][1],outpts[3][1])

    # Bringing the points to scale of (0,0)
    for i in range(len(outpts)):
        outpts[i][0]-=lowx
        outpts[i][1]-=lowy

    # Finding max of x and y-coords, in new shifted scale
    highx=max(outpts[0][0],outpts[1][0],outpts[2][0],outpts[3][0])
    highy=max(outpts[0][1],outpts[1][1],outpts[2][1],outpts[3][1])

    # Performing the transformation of image to new co-ordinates
    M = cv2.getPerspectiveTransform(inpts,outpts)
    fimg = cv2.warpPerspective(img,M,(highx,highy))

    
    fimg = fimg[fac:(width-fac), fac:(height - fac ) ]   # zoom in and crop

    fimg = cv2.resize(fimg, (output_dim, output_dim), interpolation = cv2.INTER_AREA)                       # for exact resizing  (cannot use resize_image function as it can generate non-symmetric images  )
    
    return fimg  # Transformed image is returned

def rnd_distort(coord =1, fac =10):
    return max(  (coord + int( fac *  np.sin(random.random()* np.pi * 360)) ), 1 )  ## return minimum of 1 value 
#======================================================================================================================

def randMat(M,N,P):                                                            # return randome bianry matrix of M * N shape with 1 or of probablity P # https://stackoverflow.com/questions/43065941/create-binary-random-matrix-with-probability-in-python
    a,b=   (-1.25, 1.25) 
    return np.random.choice([a,b], size=(M,N), p=[P, 1-P])                      # a,b = -0.45, 1.85  # P is the prof of "a"

'''
#testing
img = cv2.imread("logo.png")
img1 = agument_noise(img, noise_type = "random"   , mean = 5, var = 6)
img2 = agument_noise(img, noise_type = "gaussian" , mean = 5, var = 6)
cv2.imshow("noised random",img1)
cv2.imshow("noised gaussian",img2)
cv2.waitKey(0)
'''
#print(adj_screen_dim())


from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def watermark_inference(input ="test.png", output="./output.png",  text =" [CRISP inference]" , resize_flag = False , dim = (1560,390)):
    img = Image.open(input)

    if resize_flag:                           # resize the image s suggested
        img = img.resize(dim) 

    drawing = ImageDraw.Draw(img)             # make the image editable
    font = ImageFont.truetype("arial.ttf", 12)  # fot size of the result stamp    

    #text = " [ContourGANs inference]\n Source : %s\n Inference Class : %s\n Confidence : %s"%(input_image_path,input_image_path, input_image_path)   
    text_w, text_h = drawing.textsize(text, font)                           
    
    c_text = Image.new('RGB', (text_w, (text_h + 5)), color = '#000000') # +5 is added for slight lower of background view
    drawing = ImageDraw.Draw(c_text)
    
    drawing.text((0,0), text, fill="#ffffff", font=font)
    c_text.putalpha(100)
   
    img.paste(c_text, (5,5), c_text)                        #  w, h = img.size #  pos =  w - text_w, (h - text_h) - 50
    img.save(output)

