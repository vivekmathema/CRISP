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

def show_logo():
    m= "\n"
    m +=" ░▒▓██████▓▒░░ ▒▓███████▓▒░░ ▒▒▓█▓▒ ░░▒▓███████▓▒░ ▒▓███████▓▒░  ░▒▓█▓▒░ ▒▓█▓▒░" +"\n"
    m +="░▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒  ▒▓█▓▒░ ▒▒▓█▓▒ ░▒▓█▓▒░        ▒▓█▓▒░░▒▓█▓▒  ░▒▓█▓▒░ ▒▓█▓▒░ " +"\n"
    m +="░▒▓█▓▒░      ░ ▒▓█▓▒  ▒▓█▓▒░ ▒▒▓█▓▒ ░▒▓█▓▒░        ▒▓█▓▒░░▒▓█▓▒  ░▒▓█▓▒░ ▒▓█▓▒░" +"\n" 
    m +="░▒▓█▓▒░      ░ ▒▓███████▓▒░░ ▒▒▓█▓▒ ░░▒▓██████▓▒░░ ▒▓███████▓▒░  ░▒▓█▓▒░ ▒▓█▓▒░ " +"\n"
    m +="░▒▓█▓▒░      ░ ▒▓█▓▒░░▒▒▓█▓░ ▒▒▓█▓▒       ░▒▓█▓▒░▒ ▒█▓▒░         ░▒▓█▓▒░ ▒▓█▓▒░ " +"\n"
    m +="░▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░░▒▒▓█▓░ ▒▒▓█▓▒       ░▒▓█▓▒░▒ ▒█▓▒░         ░▒▓█▓▒░ ▒▓█▓▒░ " +"\n"
    m +=" ░▒▓██████▓▒░░ ▒▓█▓▒░░▒▒▓█▓░ ▒▒▓█▓▒ ░▒▓███████▓▒░░ ▒▓▓▒▒         ░▒▓█▓▒░ ▒▓█▓▒░" +"\n"
    return m

#================================================================================Librraies
import os
import platform
import subprocess
import numpy as np
import scipy as sp
import os, sys, time
import shutil
from pathlib import Path
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
from math import floor
from tqdm import tqdm
import glob
#================================================================================Pyqt5 Standard Libraries and dependent libs
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import  *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap
import pyqtgraph as pg
import pyqtgraph.exporters
#================================================================================Image utilities
from PIL import Image, ImageDraw, ImageFont
import cv2
#from swarmplot import  custom_noise
#================================================================================Core DL librearies 
import tensorflow as tf
import keras
from keras import backend as K    
from keras.layers import *
from keras.applications.inception_v3 import InceptionV3,preprocess_input
#================================================================================
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense, AveragePooling2D ,Input, Lambda
from keras.layers import Reshape, UpSampling2D, Activation, Dropout, Flatten, Conv2DTranspose
from keras.models import Model, model_from_json, Sequential , load_model
from keras.optimizers import Adam, SGD, RMSprop, Adamax, Nadam, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
#=================================================================================For Classifier
from keras.layers import Input, Lambda, Dense
from keras.layers import LeakyReLU
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.multiclass import OneVsRestClassifier
#=================================================================================For GPU memory cleaning
try:
    from numba import cuda
except:
    pass
#=================================================================================Import custom libraries
from cmd_args import *
from utils_tools import *
from hed_mode import *
from roi_selector import *
from iafg_runner import *
from autoroi import *
#==============
import warnings 
#==============
import itertools
from itertools import cycle
from colorama import *
#================== for silenseing imagio warnings (optional cosmetics module)
import imageio
import imageio.core.util
def silence_imageio_warning(*args, **kwargs):                                             # to try silcence imageio version related warnings
    pass   
imageio.core.util._precision_warn = silence_imageio_warning

#================= PYQT UI
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import  *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap
import pyqtgraph as pg
#======================Miscellillinous procedures for UI appreance
qtStyle =['Breeze', 'Oxygen', 'QtCurve', 'Windows', 'Fusion'] 
#=====================
os.chdir(os.path.dirname(os.path.realpath(__file__)))  #  change the current path for OS
#=====================                                 # Uncomment this section if your GPU to be used has index 1 (or 2,3..), instead of '0' used as default
gpu_index = "1"                                        # set GPU index (for multiple GPUs) #  1 --> RTX3070  2 --> RTX2080Ti, 0 --> RTX3090
os.environ["CUDA_VISIBLE_DEVICES"]= gpu_index          # for RTX 2080Ti == 2
#======================


#==============for plotting OneVsRest AUROCs
def plot_multi_class_auroc(true_values, predicted_values ,class_keys , show_plt= True):
    # Get the unique classes
    classes             = np.unique(true_values)                                    # holds the unique classes    
    true_binarized      = label_binarize(true_values, classes=classes)              # Binarize the true and predicted values
    predicted_binarized = label_binarize(predicted_values, classes=classes)
    auroc_scores = []
    plt.figure(figsize=(10, 8))  if show_plt == True else None

    # Calculate AUROC for each class and plot ROC curves
    print("\n[  Computed OneVsRestClassifier AUROCs  ]")
    print("       Class              |   Code |   AUROC")
    print("====================================================")
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(true_binarized[:, i], predicted_binarized[:, i])
        auroc = auc(fpr, tpr)
        auroc_scores.append(auroc)
        plt.plot(fpr, tpr, label=f'Class {class_keys[classes[i]]} (AUROC = {auroc:.4f})') if show_plt == True else None
        print(f"{class_keys[classes[i]]:<25} | {classes[i]:<5}  |  {auroc:.4f}")
    print("====================================================")
    average_auroc = np.mean(auroc_scores)                                 # Compute the average AURO    
    print(f"Average AUROC:                        {average_auroc:.4f}")   # # Print and return AUROC values, set spaces are kept for GUI

    if show_plt == True:                                                  # show midway AUROC lines
        #============== Plot settings =====================                                       
        plt.axhline(y=average_auroc, color='r', linestyle='--', label=f'Average AUROC = {average_auroc:.2f}')  # Add a line for the average AUROC
        # Plot the diagonal line for random predictions
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False positive rate', fontsize = 16)
        plt.ylabel('True positive rate', fontsize = 16)
        plt.xticks(fontsize=15) # , rotation=90)
        plt.xticks(fontsize=15)
        plt.title('receiver operating characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.grid()
        plt.show()
    return auroc_scores, average_auroc
    #=====================

def img_convert_for_inference(image_list, img_dim):
    #=================================================================
    img_data = [ cv2.imread(img) for img in image_list]                                                  # read images into an array and store it to img_data
    img_data = [ cv2.resize(img, (img_dim, img_dim)) for img in img_data   ]                              # resize all images
    img_data = [ img.astype("float")/255.0  for img in img_data  ]                                        # normalize the images array
    img_data = [ img_to_array(img) for img in img_data  ]                                                 # read image to array
    img_data = [ np.expand_dims(img, axis = 0)  for img in img_data  ]                                    # generate prediction
    #=================================================================
    return img_data      


def get_filename_and_extension(file_path):    
    filename_with_extension = os.path.basename(file_path)                                  # Extract the filename with its extension    
    filename, extension = os.path.splitext(filename_with_extension)                        # Split the filename into name and extension    
    return filename, extension

def blur_fac(img, base = 100, use_sigmoid = True):                                        # crunch the blur valeu down to 0-1 using sigmoid function on log100 of the blur vlaues
	epselon = 0.00025                                                                     # to avoid blur value getting zero which can cause error
	blur = cv2.Laplacian(img, cv2.CV_64F).var()
	try:
		blur = math.log(blur, base)
	except:
		blur = epselon                                                                
		                                                                                  # make the blur value to 0.0025 (inimum) if error 9epselon)
	func = 1/(1 + np.exp(blur)) if use_sigmoid else blur                                  # crunch it down only for the Generator in GAN
	return func

def matplot_to_cv2(img_data):                                                             # converts single matplot (RGB image to cv2 image)
	figure   = img_data
	figure   = (figure + 1) / 2 * 255
	figure   = np.round(figure, 0).astype(int)
	fig = plt.figure(1)
	fig.tight_layout()
	plt.imshow(figure.astype('uint8'))
	fig.canvas.draw()
	image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	return  image_from_plot[:, :, ::-1]


# This one works withour eror # IMPORTANT, if more than 2 classes the auroc will be zero
# https://stackoverflow.com/questions/43263111/defining-an-auc-metric-for-keras-to-support-evaluation-of-validation-dataset
def auroc(y_true, y_pred):
    try:
        auc = tf.metrics.auc(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc
    except:
        return 0.00


def write_to_file(file_path, data, header=None):
    mode = 'w' if header is not None else 'a'
    with open(file_path, mode) as file:
        print("[%s] | Multi AUROCs written to : %s"%(time_stamp(), file_path)) if mode == 'w' else None
        if header is not None:
            file.write(header + '\n')
        if data is not None:
            file.write(data +"\n") # write data ending with new line

class CustomMessageBox(QMessageBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("CRISP-II Initalization")
        self.setText("Restore recently saved configuration?")
        self.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        
        # Create two checkboxes
        self.checkbox1 = QCheckBox("Restore the last used single-pair Deepstacking ROIs (if exists)")
        self.checkbox1.setChecked(True)
        #self.checkbox2 = QCheckBox("Remember the last used multi-class Deepstacking ROIs (if exists)")
        #self.checkbox2.setChecked(True)
        
        # Create a widget to hold the checkboxes and add them to a layout
        checkbox_widget = QWidget()
        checkbox_layout = QVBoxLayout()
        checkbox_layout.addWidget(self.checkbox1)
        #checkbox_layout.addWidget(self.checkbox2)
        checkbox_widget.setLayout(checkbox_layout)
        
        # Add the checkbox widget to the message box
        self.layout().addWidget(checkbox_widget, 1, 0, 1, self.layout().columnCount())

    def checkbox_values(self):
        return self.checkbox1.isChecked() #, self.checkbox2.isChecked()


class img_generator: #Picture iterator for easy repeated calling
    
    def __init__(self, imgs, mode='gan', batch_size=4, img_dim =512 , scale_fac =1 ):
        self.imgs = imgs
        self.batch_size = batch_size
        self.mode = mode
        self.img_dim =   img_dim
        self.scale_fac = scale_fac
        if len(imgs) % batch_size == 0:
            self.steps = len(imgs) // batch_size
        else:
            self.steps = len(imgs) // batch_size + 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        X = []
        while True:
            np.random.shuffle(self.imgs)
            for i,f in enumerate(self.imgs):
                X.append(imread(f, self.mode,  self.img_dim, self.scale_fac) )
                if len(X) == self.batch_size or i == len(self.imgs)-1:
                    X = np.array(X)
                    yield X
                    X = []

class FID:
    """FID calculation based on Python
    """
    def __init__(self, x_real,from_generator=False, batch_size=None,steps=None):                             # Initialize, save the statistical results of real samples for multiple tests     
        fname ="./base_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"                     # Loading locally the weights 
        self.base_model = InceptionV3(include_top=False, weights = fname,  pooling='avg')
        self.mu_real,self.sigma_real = self.evaluate_mu_sigma(x_real, from_generator, batch_size, steps)
    
    def evaluate_mu_sigma(self, x,  from_generator=False, batch_size=None, steps=None):                      # Calculate the mean and covariance matrix from the sample

        if from_generator:
            steps = steps if steps else len(x)
            def _generator():
                for _x in x:
                    _x = preprocess_input(_x.copy())
                    yield _x
            h = self.base_model.predict_generator(_generator(),
                                                  verbose=True,
                                                  steps=steps)
        else:
            x = preprocess_input(x.copy())
            h = self.base_model.predict(x,
                                        verbose=True,
                                        batch_size=batch_size)
        mu = h.mean(0)
        sigma = np.cov(h.T)
        return mu,sigma

    def evaluate(self, x_fake, from_generator=False, batch_size=None, steps=None):        #Calculate FID value
        mu_real,sigma_real = self.mu_real,self.sigma_real
        mu_fake,sigma_fake = self.evaluate_mu_sigma(x_fake,
                                                    from_generator,
                                                    batch_size,
                                                    steps)
        mu_diff = mu_real - mu_fake
        sigma_root = sp.linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)[0]
        sigma_diff = sigma_real + sigma_fake - 2 * sigma_root
        return np.real((mu_diff**2).sum() + np.trace(sigma_diff))


def hide_tf_warnings(supress_msg = False):                  # Remove warning and error messages
	if supress_msg ==  False:
		return
	try:                                                    # try to prevent all tensorflow warning messages
		warnings.filterwarnings("ignore")
		os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
		os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
		tf.logging.set_verbosity(tf.logging.ERROR) # also tried  DEBUG, FATAL, WARN, INFO
		tf.logging.set_verbosity(tf.logging.INFO)  # also tried  DEBUG, FATAL, WARN, INFO
		tf.logging.set_verbosity(tf.logging.FATAL) # also tried  DEBUG, FATAL, WARN, INFO
	except:
		pass

def UseProcessor(use_processor, mem_growth_flag = "True"):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	if use_processor =="CPU":
		CUDA_VISIBLE_DEVICES               =  "-1"
		os.environ['CUDA_VISIBLE_DEVICES'] =  '-1'                      # aeactrivate environment for gpu
	else:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index                  # USE RTX 2080 in thsi computer which is 1
		if tf.test.gpu_device_name():                                   # Mainly use GPUs (fast)
			print('[%s] | GPU found & Activated.Running --> [GPU %s: %s]'%(time_stamp(), gpu_index, tf.test.gpu_device_name()) )
		else:
			print('[%s] | N/A or Supressed. Forced run -->  [CPU MODE]'%time_stamp())

def reset_gpu():                                            # reset GPU device and free memory
	if tf.test.gpu_device_name():
		try:
			device = cuda.get_current_device()
			device.reset()
			print("[%s] | GPU memory released..."%time_stamp())
		except:
			print("[%s] | Warning! GPU memory reset failed. May require manual program restart for proper reset"%(time_stamp()) , color = 5)
			pass
        
def set_memory_growth(): # set the GPU parameters	
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index                      # USE RTX 2080 in thsi computer which is 1
	config = tf.ConfigProto()
	config.gpu_options.allow_growth= self.gan_mem_growth_flag
	#config.gpu_options.per_process_gpu_memory_fraction = 0.5
	sess = tf.Session(config=config)
	K.set_session(sess)


from UIutils import BaseClass                                                                                   # loads teh class of variables

class Model_GAN(BaseClass):

    def explore_dir(self, dir_path ="."):
        system = platform.system()
        if system == "Windows":
            try:        
                os.startfile(dir_path ) # For Windows systems
            except:
                self.logger("[%s | Error! openining directory: %s.\nTry selecting the directory (if no set) first using GUI dialog box"%(time_stamp() , dir_path), color = 5)

        elif system == "Linux": 
            try:       
                subprocess.Popen(['xdg-open', dir_path ]) # For Linux systems
            except:
                self.logger("[%s | Error! openining directory: %s.\nTry selecting the directory (if no set) first using GUI dialog box"%(time_stamp() , dir_path), color = 5)

        elif system == "Darwin":
            try:        
                subprocess.Popen(['open', folder_path])  # For macOS systems
            except:
                self.logger("[%s | Error! openining directory: %s.\nTry selecting the directory (if no set) first using GUI dialog box"%(time_stamp() , dir_path), color = 5)
        else:
            print(f"Unsupported operating system to view directory: {system}")

    def set_ui_stayontop(self, opaque_val = 100):
        # Change window flags to remove 'Stay on Top' and return to default
        if self.actionStay_on_Top.isChecked():
            self.setWindowFlags(Qt.WindowStaysOnTopHint)
            self.logger("[%s] | CRISP GUI  stay on top mode activated"%time_stamp() , color = 3  )
        else:
            self.setWindowFlags(~Qt.WindowStaysOnTopHint)
            self.logger("[%s] | CRISP GUI  stay on top mode deactivated"%time_stamp() , color = 3)

    def show_message_box(self):
        msg_box = CustomMessageBox()
        response = msg_box.exec_()
        
        if response == QMessageBox.Yes:
            #checkbox1_state, checkbox2_state = msg_box.checkbox_values()  # for future in more than 1 needed
            checkbox1_state  = msg_box.checkbox_values()
            self.load_config(config_fname = "./config/settings.conf",  msg = "")   # load last saved configuations
            self.load_ssim_data(show_msg = False) if checkbox1_state else None
  
        else:
            print("[%s] | Loading default configurations"%time_stamp()) 
            #self.load_config(config_fname = "./config/default.conf",  msg = "")  # only load default configuations
            

    def prediction_heatmap(self, img = None, model = None, target_layer_index = 0, show_model_summary = False, out_fname = None):  # target_layer = 'block5_conv2'  # lasy layer for same dim
        self.update_vars()
        def preprocess_image(img_path, target_size=(512, 512)):
            img = image.load_img(img_path, target_size=target_size)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            return img

        if self.run_mode !=  "classifier_inference":
            return

    
        #model = load_model("E:/GRASP_root/classifier_data/trained_weights/contour_features_model_dsCRISP2-3-4AF1THSI_Disease_512pxl_VGG16.h5") # Load the pre-trained model
        if show_model_summary:
            print("Model layers summary:\n",model.summary() ,"\n")
    
        blockX_convY = ["block3_conv3", "block3_conv2","block4_conv3","block4_conv2", "block5_conv3",  "block5_conv2"]   # corrsponding levels
        target_layer = blockX_convY[target_layer_index]                                                                  # get the current index
    
        target_layer_output = model.get_layer(target_layer).output                # Get the output of the block5_conv3 layer
        activation_model = Model(inputs=model.input, outputs=target_layer_output) # Create a new model that takes an input image and outputs the activations of block5_conv3
    
        # Load and preprocess your image
        #img = preprocess_image(img_path, target_size=(512, 512))
        
        activations = activation_model.predict(img)        # Get the activations of block5_conv3 for the input image        
        heatmap = np.maximum.reduce(activations, axis=-1)  # Get the maximum value across channels for each spatial location
        # Normalize the heatmap
        heatmap /= np.max(heatmap)
        # Plot the heatmap
        plt.close('all')
        heat_fig = plt.gca()
        frame1 = plt.gca()
        heat_fig.axes.xaxis.set_ticklabels([])
        heat_fig.axes.yaxis.set_ticklabels([])
        plt.title(f"Inference heatmap @ {target_layer}")
        plt.imshow(heatmap[0], cmap='viridis')
        plt.colorbar(label='Prediction-contributing zones')
        plt.savefig(out_fname) if out_fname != None else None
        self.logger("Heatmap written to : %s"%out_fname, color = 1) if out_fname != None else None
        plt.show()
        plt.close() if not self.pause_each_heatmap else None


    def classification_analysis(self, true_class, pred_class):
        # Convert to numpy arrays
        true_class = np.array(true_class)
        pred_class = np.array(pred_class)
        
        cm = confusion_matrix(true_class, pred_class) # Calculate confusion matrix
    
        # make Classification report
        class_report    = classification_report(true_class, pred_class, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()

        # Accuracy
        accuracy = accuracy_score(true_class, pred_class)
    
        # Precision, Recall, F1-Score (weighted)
        precision, recall, f1, _ = precision_recall_fscore_support(true_class, pred_class, average='weighted')
    
        self.logger("\nBasic statistics on inference results:\n", color =3)
        self.logger("--------------------------------------------")
        self.logger("\nConfusion matrix:\n", color = 4)
        self.logger(cm, color = 3)
        self.logger("\nClassification report:", color =4)
        self.logger(class_report_df, color = 3)
        self.logger(f"\nOverall accuracy  : {accuracy:.2f}" ,color =4   if precision > 0.5 else  5)
        self.logger(f"Weighted Precision: {precision:.2f}", color = 4 if precision > 0.5 else  5)
        self.logger(f"Weighted Recall   : {recall:.2f}", color =4     if precision > 0.5 else  5)
        self.logger(f"Weighted F1 Score : {f1:.2f}", color =4         if precision > 0.5 else  5)
        self.logger("--------------------------------------------")
    
        # Plotting confusion matrix
        plt.close("all")
        plt.figure(figsize=(12, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(true_class), yticklabels=np.unique(true_class))
        plt.title('Confusion Matrix' , fontsize =14)
        plt.xlabel('Predicted Classes', fontsize =14)
        plt.ylabel('True Class' , fontsize =14)
        # Adjust layout to ensure everything fits
        plt.tight_layout()
        plt.show()


    def clear_terminal_view(self):
        if self.qm.question(self,'CRISP',"Clear Terminal/Console view?\nCaution: Please select proper Model ID from model id listview if needed", self.qm.Yes | self.qm.No) == self.qm.No:
            return
        try:
            os.system('cls' if os.name=='nt' else 'clear')
            print(Style.BRIGHT + Fore.BLUE + show_logo())
            self.logger("[%s] | Terminal | Console view cleared successfully...\n"%time_stamp() , color =3)
        except:
            self.logger("[%s] | Warning! Failed to clear Terminal | Console view..."%time_stamp() , color =5)
            pass


    def apply_mem_growth(self):
    	self.logger("[%s] | Setting memory growth if GPU is being used..."%start_time())
    	try:
    		set_memory_growth() if self.gan_gpu_mem_growth and (self.run_mode == "train_gan" or self.run_mode == "synthesis" ) else None
    		set_memory_growth() if self.cls_gpu_mem_growth and (self.run_mode == "classifier_training"  or  self.run_mode ==  "classifier_inference") else None
    		set_memory_growth() if (self.run_mode == "train_sr"  or  self.run_mode == "SR Upscaling") else None
    	except:
    		pass


    def set_muticlass_source_dataset(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if fname:
            self.multiclass_source_dataset.setText(fname)                                        # Set Dirname as path where images are located
        else:
            None

    def set_roids_path(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if fname:
            self.roids_src_path.setText(set_dirpath(fname) )                                        # Set Dirname as path where images are located
        else:
            None

    def update_roids_fname(self):
    	if self.roi_ds_id.toPlainText().count("\n"):                                        # show console message only if hit inter to confrim
    		self.logger("[%s] | ROI & Deep stacking datafile ID      : %s"%(time_stamp(), self.roi_ds_id.toPlainText().strip() + ".data") , color = 4)
    		self.logger("[%s] | Updated ROI & Deep stacking file path: %s"%(time_stamp(), self.ds_pos_filepath.toPlainText()) , color = 4)
    	pass 


    def set_gan_online_model_filepath(self):                                                 # for onlien GAN model selection path
        folderName = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if folderName:
            self.gan_online_model_selected.setText(set_dirpath(folderName) )
        else:
            None   


    def set_cls_online_model_filepath(self):                                                 # for onlien GAN model selection path
        folderName = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if folderName:
            self.cls_online_model_selected.setText(set_dirpath(folderName) )
        else:
            None   

    def set_classifier_source_data(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if fname:
            self.classifier_source_path.setText(fname)
            self.classifier_model_id.setText(os.path.basename(fname) )
        else:
            None    	

    def set_classifier_model_path(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if fname:
            self.classifier_model_path.setText( set_dirpath(fname) )
        else:
            None
        self.cls_model_pushbutton.setChecked(True)                              
        self.show_cls_model_list()


    def set_classifier_test_image_path(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if fname:
            self.classifier_testing_path.setText(fname)
        else:
            None   

    def set_classifier_report_path(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if fname:
            self.classifier_report_path.setText( set_dirpath(fname) )
        else:
            None   

    def set_classifier_output_path(self):
    	fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))
    	if fname:
    		self.classifier_output_path.setText( set_dirpath(fname) )
    	else:
    		None   

    def set_classifier_src_path(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))
        if fname:
            self.cls_src_dataset_path.setText( set_dirpath(fname) )
            self.cls_dataset_class.setText(os.path.basename(fname))
        else:
            None  


    def set_classifier_entire_src_path(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))
        if fname:
            self.cls_entire_path.setText(set_dirpath(fname) )
            self.cls_dataset_name.setText(os.path.basename(fname)) 
        else:
            None  


    def set_classifier_dst_path(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))
        if fname:
            self.cls_dst_dataset_path.setText( set_dirpath(fname) )
        else:
            None  

    def set_pred_heatmap_store_path(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))
        if fname:
            self.cls_store_pred_heatmap_path.setText(fname)
        else:
            None  


    def set_multiclass_val_cohort_path(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))
        if fname:
            self.multiclass_val_cohort_path.setText(fname)  
            self.multiclass_val_cohort.setText(os.path.basename(fname))
        else:
            None  

    #-----------------------------Setting filepath for Countour A  deepstacking and counter B Deepstacking

    def set_contourA_dsimg_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select Class-A Representative AFRC image file(.jpg) ","","All Files (*.*);;Image Files (*.jpg, *.png, *.jpeg)", options=options)
        if fileName:
            self.contourA_dspath.setText (fileName)
            self.contourA_dspath_ds.setText(os.path.dirname(os.path.dirname(fileName)))  # set it to the two step inside  CLASS/afrc_image/... to CLASS/
            self.ds_pos_filepath.setText(os.path.join("./roi_data", "ds_roi_"+ os.path.basename(self.contourA_dspath_ds.toPlainText())+ ".data" ))
        else:
            None

    def set_contourB_dsimg_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select Class-B Representative AFRC image file(.jpg) ","","All Files (*.*);;Image Files (*.jpg, *.png, *.jpeg)", options=options)
        if fileName:
            self.contourB_dspath.setText (fileName)
            self.contourB_dspath_ds.setText(os.path.dirname(os.path.dirname(fileName))) # set it to the two step inside  CLASS/afrc_image/... to CLASS/        
        else:
            None

    #-----------------------------Dialogue BoX
    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open source image file(.jpg) ","","All Files (*.*);;Image Files (*.jpg, *.png, *.jpeg)", options=options)
        
        return fileName                             # Save file dialogue box  block

    def saveFileDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save CRISP results","","All Files (*);;Text Files (*.txt)", options=options)
        return fileName                              # Open file dialogue box

    def set_ds_restore_filepath(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select file to restore DeepStacked and ROIs","","All Files (*.*);;SSIM and ROIs data files (*.data)", options=options)
        if fileName:
            self.ds_pos_filepath.setText(fileName)
        else:
            None

    def set_model_folder(self):                                # for storing result output images from GAN
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if fname:
            self.gan_model_path.setText(fname )
        else:
            None
        self.gan_model_pushbutton.setChecked(True)             # set model path and display existing models
        self.show_gan_model_list()

    def set_result_folder(self):                               # for storing result output images from GAN
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if fname:
            self.gan_output_path.setText(fname) 
        else:
            None

    def set_preview_folder(self):                                     # for storing result output images from GAN
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if fname:
            self.preview_imgs_path.setText(fname) #set_dirpath (for adding slash in last of file name)
        else:
            None           

    def set_data_image(self):
        sourcePath = str(QFileDialog.getExistingDirectory(self, os.path.join(os.getcwd(), "classifier_data/datasets")))
        self.gan_src_image.setText(sourcePath) if sourcePath  != "" else None
        self.gan_processed_img_path.setText(sourcePath) if sourcePath   != "" else None         # automatically put the processed image path
        self.org_dim_rois_extract.setText(sourcePath) if sourcePath   != "" else None           # automatically set the processing path for orgdim
        self.model_id_box.setText(os.path.basename(sourcePath)) if sourcePath   != "" else None # autoset gan model name ID

    def set_processed_image_path (self):
        sourcePath = str(QFileDialog.getExistingDirectory(self, os.getcwd())) 
        self.gan_processed_img_path.setText(sourcePath) if sourcePath   != "" else None   	

    def set_ContourA_img(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select Class-A Contour image file(.jpg) ","","All Files (*.*);;Image Files (*.jpg, *.png, *.jpeg)", options=options)
        if fileName:
            self.contourA_imagepath.setText (fileName)
        else:
            None

    def set_pth_multiclass_src_dataset(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if fname:
            self.ds_src_dataset_location.setText( set_dirpath(fname) )            
        else:
            None  

    def set_pth_multiclass_control_group(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if fname:
            self.ds_src_control_id.setText(os.path.basename(fname))
        else:
            None  

    #============================================================= for multiclass_deepstacking

    def set_multiclass_ds_dataset(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if fname:
            self.multiclass_contourA_dspath.setText(os.path.basename(fname) )
            self.multiclass_contourA_fpath.setText(fname)            
        else:
            None  

    def set_multiclass_ds_control_group(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))       
        if fname:
            self.multiclass_control_group_fpath.setText(fname)
            self.multiclass_control_group_dspath.setText(os.path.basename(fname))
        else:
            None  

    def set_multiclass_ds_pos_fname(self):
        fname = str(QFileDialog.getExistingDirectory(self, os.getcwd()))   
        if fname:    
            self.multiclass_ds_pos_dspath.setText(os.path.basename(fname))
            self.multiclass_ds_pos_fpath.setText(os.path.join(fname, "rois_ds_info.data"))
        else:
            None 
    #======================================================================================= 
              
    def set_random_afrc_images(self):
    	self.set_updated_gui_opts()
    	if self.qm.question(self,'AFRC',"If Random selection is choosen, this will randomly select one image for each class and ignore AFRC image [not recommended]. Continue?" , 
    		                self.qm.Yes | self.qm.No) == self.qm.No:
    		self.logger("[INFO] Random AFRC image selection skipped....")
    		return
    	if self.iafg_rnd_select:                                        # update the GUI prameters
    		self.contourA_dspath.setText( random.choice(get_source_images(os.path.dirname(self.contourA_dspath.toPlainText()))))
    		self.contourB_dspath.setText( random.choice(get_source_images(os.path.dirname(self.contourB_dspath.toPlainText()))))


    def set_ContourB_img(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select Class-B Contour image file(.jpg) ","","All Files (*.*);;Image Files (*.jpg, *.png, *.jpeg)", options=options)
        if fileName:
            self.contourB_imagepath.setText (fileName)
        else:
            None

    #-----------------------------| Select configuration file for the savined configuration
    def select_custom_configfile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select constom configuration file ","./config/","All Files (*.*);;Image Files (*.conf)", options=options)
        if fileName:
        	self.load_config(config_fname = fileName,  msg = "Load custom configuration?")
        else:
            None

    #============== MULTICLASS ROIs

    def preview_multiclass_folder_list(self):
        # Get the directory path from the QLineEdit
        directory_path = self.ds_src_dataset_location.toPlainText()
        control_group  = self.ds_src_control_id.toPlainText()

        if  directory_path.strip() == "" or control_group.strip()=="":
            msg ="ERROR! Database directory or Control name not supplied"
            self.qm.about(self, "CRISP v0.2", msg)
            return         

        if os.path.isdir(directory_path):
            all_items = os.listdir(directory_path)                                                              # Get all items in the directory     
            folders = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]         # Filter out only the directories
            self.multiclass_afrc_path     = [ "/".join([directory_path,each_class]) for each_class in folders ] # set list for the multiAFRC proicessing
            self.multiclass_control       =   "/".join([directory_path,control_group])                          # set control group for multi afrc processing

            self.logger("[%s] | Number of Classes or groups"%time_stamp(), color =2)
            self.logger("--------------------------------------")
            for index,each_class in enumerate(self.multiclass_afrc_path):
                self.logger(f"{index} | {each_class}", color =3)
            self.logger("--------------------------------------")
            
            
            # Display the folder names in the QTextEdit
            self.multiclass_afrc_groups_list.clear()
            self.multiclass_afrc_groups_list.addItem("Dataset path: %s"%directory_path)
            self.multiclass_afrc_groups_list.addItem("----------------------------------")
            for folder in folders:                                                                          # Show folder and control group
                self.multiclass_afrc_groups_list.addItem(f"{folder:<70} ")
            self.multiclass_afrc_groups_list.addItem("----------------------------------")       
        else:
            self.multiclass_afrc_groups_list.clear()
            self.multiclass_afrc_groups_list.addItem("Invalid directory path. Please enter a valid path.")
            

    def multiclass_afrc_groups_list_item_clicked(self, item):
        # Get the text of the clicked item
        folder_name = item.text()
        # Print the selected folder name to the console        
        if folder_name.count("|") >0:                         # check if it's a pair
            self.current_afrc_class.setText(folder_name)
            print(f"Selected Folder: {folder_name}")


    def preview_multiclass_deepstacking_folder_list(self):
        # Get the directory path from the QLineEdit
        directory_path   = self.multiclass_contourA_fpath.toPlainText()
        control_group    = self.multiclass_control_group_fpath.toPlainText()
        control_group_id = self.multiclass_control_group_dspath.toPlainText()

        if  directory_path.strip() == "" or control_group.strip()=="":
            msg ="ERROR! Database or Control Group name not supplied"
            self.qm.about(self, "CRISP v0.2", msg)
            return         

        if os.path.isdir(directory_path):
            all_items = os.listdir(directory_path)                                                              # Get all items in the directory     
            folders = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]         # Filter out only the directories
            self.deepstack_multiclass_path     = [ "/".join([directory_path,each_class]) for each_class in folders ] # set list for the multiAFRC proicessing
            self.deepstack_multiclass_control  =   "/".join([directory_path,control_group_id])                          # set control group for multi afrc processing

            self.logger("[%s] | Number of Classes or groups"%time_stamp(), color =2)
            self.logger("--------------------------------------")
            for index,each_class in enumerate(self.deepstack_multiclass_path):
                self.logger(f"{index} | {each_class}", color =3)
            self.logger("--------------------------------------")
            
            # Display the folder names in the QTextEdit
            self.multiclass_deepstack_groups_list.clear()
            self.multiclass_deepstack_groups_list.addItem("Dataset path: %s"%directory_path)
            self.multiclass_deepstack_groups_list.addItem("----------------------------------")
            for folder in folders:
                if folder != control_group_id:                                                                   # Show folder and control group
                    self.multiclass_deepstack_groups_list.addItem(f"{folder:<30} | {control_group_id:<30} ")
            self.multiclass_deepstack_groups_list.addItem("----------------------------------")       
        else:
            self.multiclass_deepstack_groups_list.clear()
            self.multiclass_deepstack_groups_list.addItem("Invalid directory path. Please enter a valid path.")


    def multiclass_deepstacking_groups_list_item_clicked(self, item):
        # Get the text of the clicked item
        folder_name = item.text()
        control_droup_id = self.multiclass_control_group_dspath.toPlainText()
        # Print the selected folder name to the console        
        if folder_name.count("|") >0:                         # check if it's a pair
            self.current_ds_class.setText(folder_name)
            print(f"{folder_name:<30}")



    #========[ROI BLOCKS]===============================

    def set_orgdim_rois_path(self):
        sourcePath = str(QFileDialog.getExistingDirectory(self, os.getcwd())) 
        self.org_dim_rois_extract.setText(sourcePath) if sourcePath   != "" else None   

    def auto_compute_ssim_roi(self):
    	if self.qm.question(self,'CRISP',"Auto compute the single class pair-wise ROIs?", self.qm.Yes | self.qm.No) == self.qm.No:
    		self.logger("[INFO] Single class Pair-wise ROIs operation skipped....")
    		return
    	self.update_vars()
    	self.set_updated_gui_opts()
    	self.show_info()
    	hide_tf_warnings(supress_msg = True)

    	self.logger("#Contour imageA path           :"+ self.contourA_IAFGpth  )                                   # path for contour image class-A deepstacking
    	self.logger("#Contour imageB path           :"+ self.contourB_IAFGpth  )                                   # path for contour image class-B Deep stacking

    	if self.stacking_flag:                                                                                     # must be before the 'auto' assignment
    		num_roi = self.win_segment

    	if self.auto_roi_para_flag:
    		self.winsize          = "auto"
    		self.win_segment      = "auto"
    		self.win_shift        = "auto"
    		self.ssim_thresthold  = "auto"   # 0.95

    	self.Contour_classA = cv2.imread(self.contourA_IAFGpth  )
    	self.Contour_classB = cv2.imread(self.contourB_IAFGpth  )

    	if self.sharpen_autoroi:
    		for x in range(2):
    			self.Contour_classA = edge_enhance(self.Contour_classA)
    			self.Contour_classB = edge_enhance(self.Contour_classB)

    	if self.denoise_autoroi:
    		for x in range(2):
    			self.Contour_classA = rnd_denoise(self.Contour_classA)
    			self.Contour_classB = rnd_denoise(self.Contour_classB)


    	if self.store_plot:
    		self.plot_fname = os.path.join(self.contourA_dspath_ds.toPlainText() , "afrc_image", "score_plot.png")
            #check_dir(self.plot_fname)
    	else:
    		self.plot_fname = None

    	self.max_diff          = scan_similarity(self.Contour_classA, self.Contour_classB,       # Function RETURNS A  list: (scan_step, (win_start, win_end), SSIM_score, len(cnts), winsize)
    	                            ssim_thresthold = self.ssim_thresthold  ,                    # sets the minimum SSIM_thresthold valuie required to be in list
	                        		winsize         = self.winsize     ,                          # window size, if None, then it will be auto
	                        		win_shift       = self.win_shift   ,                          # window shifting pixels, Typical value 10 worked for me
	                        		win_segment     = self.win_segment ,                          # numbers of segmenst to device the whole image
	                        		show_scan       = self.show_scan   ,                          # flag that shows scan on screen
	                        		show_ssim_graph = self.show_scan,                             # shows the final SSIM_graph for the scann results
	                        		SSIM_mode       = self.ssim_mode,                             # runs in SSIM mode "Simple"  or in vgg16 
	                        		avoid_overlap   = self.avoid_overlap,                         # avoids overallping of teh window shift
                                    store_graph     = self.plot_fname)                            # full pathname of the file name             

	                        		                                                              
    	if self.max_diff != None:
    		self.SSIM_list.clear()
    		self.ssim_list_data =[]
    		for index, x in enumerate(self.max_diff):                                              # get index and vlaues ( no need using count as index is used)info =
    			info = str(index) +"	" + str("%0.8f"%self.max_diff[index][2]) + "	" + str(self.max_diff[index][1])
    			self.ssim_list_data.append(info) 
    			self.SSIM_list.addItem(info)
    			self.logger( str(index-1) +"	"+ str("%0.8f"%self.max_diff[index-1][2]) + "	" + str(self.max_diff[index-1][1]) ) # important it's confusing part of indexing
    			self.winsize  = self.max_diff[index-1][1][1] - self.max_diff[index-1][1][0]        # self.wisize is "auto" so need real value    			                                       
    		self.SSIM_list.setCurrentRow(0)                                                        # set the default index to zero

    		self.set_SSIM_value()                                                                  # do not autoshow the list preview during scan fill up the list for SSIM and hereafter the selection will be carried out there
    		#print(self.ssim_list_data)                                                            # print teh similarity thresthold 

    	else:
    		self.logger("\n[INFO] No difference(s) found....")
    		return

    	if self.stacking_flag and len(gval(self.max_diff)) > 0:                                    # Proceed to DeepStacking of image if any, gval is to avoid erro due to None ==0 type
            self.logger("===================================\n")
            self.logger("[%s] | Detecting DeepStacking features\n"%(time_stamp()))
            self.hstack_rois(unit_ws = self.winsize, num_roi = min(5,len(self.max_diff)))          # unit winsize and take minimum of  5 or less than 5


    	if True:                                                                                   # Store the ROIs data
            self.store_ssim_pos()

    	self.logger("\n[%s] | Computation for Deepstacking ROIs comepleted"%time_stamp(), color =4)


    #==================================================================================================

    def multiclass_auto_compute_ssim_roi(self, show_msg = True):
        if show_msg:
            if self.qm.question(self,'CRISP-II',"Compute multi-class dataset?", self.qm.Yes | self.qm.No) == self.qm.No:
                return

        self.update_vars()
        self.set_updated_gui_opts()
        self.show_info()
        hide_tf_warnings(supress_msg = True)


        self.multiclass_contourA_IAFGpth = os.path.join(self.multiclass_contourA_fpath.toPlainText(),\
                                             self.multiclass_ds_pos_dspath.toPlainText(),"afrc_image", "afrc_image.jpg")           # for normal
        self.multiclass_contourB_IAFGpth = os.path.join(self.multiclass_control_group_fpath.toPlainText(), "afrc_image", "afrc_image.jpg")    # for control group


        self.logger("#Contour imageA path           :"+ self.multiclass_contourA_IAFGpth  )                                   # path for contour image class-A deepstacking
        self.logger("#Contour imageB path           :"+ self.multiclass_contourB_IAFGpth  )                                   # path for contour image class-B Deep stacking

        if self.multiclass_stacking_flag:                                                                    # must be before the 'auto' assignment
            num_roi = self.multiclass_win_segment

        if self.multiclass_auto_roi_para_flag:
            self.multiclass_winsize          = "auto"
            self.multiclass_win_segment      = "auto"
            self.multiclass_win_shift        = "auto"
            self.multiclass_ssim_thresthold  = "auto"   # 0.95

        self.multiclass_Contour_classA = cv2.imread(self.multiclass_contourA_IAFGpth)
        self.multiclass_Contour_classB = cv2.imread(self.multiclass_contourB_IAFGpth)

        if self.sharpen_autoroi:
            for x in range(2):
                self.multiclass_Contour_classA = edge_enhance(self.multiclass_Contour_classA)
                self.multiclass_Contour_classB = edge_enhance(self.multiclass_Contour_classB)

        if self.denoise_autoroi:
            for x in range(2):
                self.multiclass_Contour_classA = rnd_denoise(self.multiclass_Contour_classA)
                self.multiclass_Contour_classB = rnd_denoise(self.multiclass_Contour_classB)

        if self.multiclass_store_plot:
            self.multiclass_plot_fname = os.path.join(os.path.dirname(self.multiclass_ds_pos_fpath.toPlainText()) , "afrc_image", "score_plot.png")
        else:
            self.multiclass_plot_fname = None


        if self.run_mode == "multiclass_deepstacking" and self.multiclass_pause_each_afrc == False:    # skip the graph if the  pause after each AFRC is deactivated

            if self.multiclass_show_scan_roi_flag.isChecked():
                self.multiclass_show_scan_roi_flag.setChecked(False)                                  # Overwrite the Show Graph option to Fase
                self.multiclass_show_scan == False
                self.pause_each_afrc = False
                self.logger("[%s] | Show ROI Graph option is overwritten to continue multiclass deepstacking"%time_stamp(), color = 5)

        
        self.multiclass_max_diff           = scan_similarity(self.multiclass_Contour_classA, self.multiclass_Contour_classB, 
                                               ssim_thresthold = self.multiclass_ssim_thresthold  ,                        # sets the minimum SSIM_thresthold valuie required to be in list
                                               winsize         = self.multiclass_winsize     ,                             # window size, if None, then it will be auto
                                               win_shift       = self.multiclass_win_shift   ,                             # window shifting pixels, Typical value 10 worked for me
                                               win_segment     = self.multiclass_win_segment ,                             # numbers of segmenst to device the whole image
                                               show_scan       = self.multiclass_show_scan,                                # flag that shows scan on screen
                                               show_ssim_graph = self.multiclass_preview_auto_roi,                         # shows the final SSIM_graph for the scann results
                                               SSIM_mode       = self.multiclass_ssim_mode,                                # runs in SSIM mode "Simple"  or in vgg16 
                                               avoid_overlap   = self.multiclass_avoid_overlap,                            # sets weatehr to allow window overlapping duing scan or stacking                                 
                                               store_graph     = self.multiclass_plot_fname)                               # RETURNS A  list: (scan_step, (win_start, win_end), SSIM_score, len(cnts), winsize)
        if self.multiclass_max_diff != None:
            self.multiclass_SSIM_list.clear()
            self.multiclass_ssim_list_data =[]
            for index, x in enumerate(self.multiclass_max_diff):                                                    # get index and vlaues ( no need using count as index is used)info =
                info = str(index) +"    " + str("%0.8f"%self.multiclass_max_diff[index][2]) + "    " + str(self.multiclass_max_diff[index][1])
                self.multiclass_ssim_list_data.append(info) 
                self.multiclass_SSIM_list.addItem(info)
                self.logger( str(index-1) +"    "+ str("%0.8f"%self.multiclass_max_diff[index-1][2]) + "   " + str(self.multiclass_max_diff[index-1][1]) ) # important it's confusing part of indexing
                self.multiclass_winsize  = self.multiclass_max_diff[index-1][1][1] - self.multiclass_max_diff[index-1][1][0]                   # self.wisize is "auto" so need real value                                                      
            self.multiclass_SSIM_list.setCurrentRow(0)                                                          # set the default index to zero

            self.set_multiclass_SSIM_value()                                                                    # do not autoshow the list preview during scan fill up the list for SSIM and hereafter the selection will be carried out there

        else:
            self.logger("\n[INFO] No difference(s) found....")
            return

        if self.multiclass_stacking_flag and len(gval(self.multiclass_max_diff)) > 0:                           # Proceed to DeepStacking of image if any, gval is to avoid erro due to None ==0 type
            self.logger("===================================\n")
            self.logger("[%s] | Detecting DeepStacking features\n"%(time_stamp()))
            self.multiclass_hstack_rois(unit_ws = self.multiclass_winsize, num_roi = min(5,len(self.multiclass_max_diff))) # unit winsize and take minimum of  5 or less than 5

        if True:                                                                                               # Store the ROIs data
            self.multiclass_store_ssim_pos()



    #====================================================================================================================================


    def hstack_rois(self, unit_ws, num_roi = 5):                                                   # horizontal stack the top_n ROIs and show teh stacked image as preview
        global stacked_array
        self.set_updated_gui_opts()
        img_height = self.Contour_classA.shape[0]    # ImgA or imgB as both must be of  same dimension
        img_width  = num_roi * unit_ws
        self.logger("--> Image height,width, unit winsize : (%d, %d, %d) "%(img_height , img_width, unit_ws) )
        self.logger("--> Total segmentes to deep stack    : %d"% num_roi)
        imgA_array, imgB_array =[],[]

        for index in range(num_roi):
            self.logger("--> Scanning position: %d - %d" % (self.max_diff[index][1][0], self.max_diff[index][1][1]) )
            imgA_array.append(self.Contour_classA[: , self.max_diff[index][1][0]: self.max_diff[index][1][1] ])
            imgB_array.append(self.Contour_classB[: , self.max_diff[index][1][0]: self.max_diff[index][1][1] ])
            self.deep_stack.append((self.max_diff[index][1][0], self.max_diff[index][1][1]) )     # record deepStack array array of position to be extracted for stacking image

        stack_imgA, stack_imgB = cv2.hconcat(imgA_array), cv2.hconcat(imgB_array)                 #  Again concatinate to diaplay
        stacked_array = self.deep_stack                                                           # share ithe sacked array so that it is preserved even though the update var is used
        contA = cv2.copyMakeBorder(stack_imgA,10,10,10,10,cv2.BORDER_CONSTANT,value= [0,0,0])     #  Order colour value = black [0,0,0]
        contB = cv2.copyMakeBorder(stack_imgB,10,10,10,10,cv2.BORDER_CONSTANT,value= [0,0,0])
        self.logger("--> Output stacked image shape (Width, Height,channel) :%d, %d, %d"%stack_imgA.shape)
        self.logger("===================================\n")
        
        if self.preview_auto_roi:
            cv2.imshow("DeepStacked image: TOP: Class-A | BOTTOM: Class-B | Segments used: %d"%self.win_segment, cv2.vconcat((contA, contB)))
            cv2.waitKey(0)  

    # for multiclass stacking info
    def multiclass_hstack_rois(self, unit_ws, num_roi = 5):                                                   # horizontal stack the top_n ROIs and show teh stacked image as preview
        global multiclass_stacked_array
        self.set_updated_gui_opts()
        img_height = self.multiclass_Contour_classA.shape[0]    # ImgA or imgB as both must be of  same dimension
        img_width  = num_roi * unit_ws
        self.logger("--> Image height,width, unit winsize : (%d, %d, %d) "%(img_height , img_width, unit_ws) )
        self.logger("--> Total segmentes to deep stack    : %d"% num_roi)
        imgA_array, imgB_array =[],[]

        for index in range(num_roi):
            self.logger("--> Scanning position: %d - %d" % (self.multiclass_max_diff[index][1][0], self.multiclass_max_diff[index][1][1]) )
            imgA_array.append(self.multiclass_Contour_classA[: , self.multiclass_max_diff[index][1][0]: self.multiclass_max_diff[index][1][1] ])
            imgB_array.append(self.multiclass_Contour_classB[: , self.multiclass_max_diff[index][1][0]: self.multiclass_max_diff[index][1][1] ])
            self.multiclass_deep_stack.append((self.multiclass_max_diff[index][1][0], self.multiclass_max_diff[index][1][1]) )     # record deepStack array array of position to be extracted for stacking image

        stack_imgA, stack_imgB = cv2.hconcat(imgA_array), cv2.hconcat(imgB_array)                                       #  Again concatinate to diaplay
        multiclass_stacked_array = self.multiclass_deep_stack                                                           # share ithe sacked array so that it is preserved even though the update var is used
        contA = cv2.copyMakeBorder(stack_imgA,10,10,10,10,cv2.BORDER_CONSTANT,value= [0,0,0])                           #  Order colour value = black [0,0,0]
        contB = cv2.copyMakeBorder(stack_imgB,10,10,10,10,cv2.BORDER_CONSTANT,value= [0,0,0])

        self.logger("--> Output stacked image shape (Width, Height,channel) :%d, %d, %d"%stack_imgA.shape)
        self.logger("===================================\n")

        if self.multiclass_pause_each_afrc:
            cv2.imshow("Deepstacked ROIs image: TOP: Class-A | BOTTOM: Class-B | Segments used: %d"%self.multiclass_win_segment, cv2.vconcat((contA, contB)))         
            cv2.waitKey(0)
        else:
            cv2.waitKey(2000)                                                                                           # just shpw for 3 seconds and continue                                                         
                        

    def make_stacked_dataset(self):
        global stacked_array
        if self.qm.question(self,'CRISPII',"Make single pair DeepStacked dataset?", self.qm.Yes | self.qm.No) == self.qm.No:
            return
        self.set_updated_gui_opts()
        self.init_logger()

        stacked_array =self.deep_stack                                                                                 # restore stack array values                                                                     
        if stacked_array == None or len(stacked_array) == 0:
            self.logger ("[%s] | Error! No stacking array detected. Use Computue ROI with DeepStacking option Enabled"%time_stamp , color =5)       # check si teh stackinmg array is created
            return 
        
        #DeepStacking for CLass-A
        img_dir =self.contourA_dspath_ds.toPlainText()                                                                 # path of teh deep stacking image
        ClassA_imgList = get_source_images(img_dir)                                                                    # get list of all supported images in CLass A folder
        self.logger("[%s] | Building DeepStacked images for Class-A"%(time_stamp()))         
        check_dir(img_dir  +"/stacked_images") 

        for raw_img in tqdm(ClassA_imgList):
            img =   stacked_img(cv2.imread(raw_img), stacking_array = stacked_array)
            fnam =  img_dir  + "/stacked_images/stacked_" + os.path.basename(raw_img)
            cv2.imwrite(fnam, img)
        
        self.logger("\n[%s] | Dataset written to: %s/stacked_images"%(time_stamp(), img_dir ))

        #Deep Stacking For Class B                            
        img_dir =self.contourB_dspath_ds.toPlainText()    
        ClassB_imgList = get_source_images(img_dir)                                                                     # get list of all supported images in CLass B folder              
        self.logger("[%s] | Building DeepStacked images for Class-B"%(time_stamp()))    
        check_dir(img_dir +"/stacked_images") 
        
        for raw_img in tqdm(ClassB_imgList):
            img =  stacked_img(cv2.imread(raw_img), stacking_array = stacked_array)
            fnam = img_dir +"/stacked_images/stacked_" + os.path.basename(raw_img)
            cv2.imwrite(fnam, img)
        
        self.logger("\n[%s] | Dataset written to: %s/stacked_images"%(time_stamp(),img_dir) )



    def multiclass_make_stacked_dataset(self):
        global multiclass_stacked_array
        if self.qm.question(self,'CRISP-II',"Compute and build multi-class DeepStacked dataset?", self.qm.Yes | self.qm.No) == self.qm.No:
            return
        self.set_updated_gui_opts()
        self.init_logger()
        self.run_mode = "multiclass_deepstacking"

        #======================= Set up the directory path for contro land disease

        directory_path   = self.multiclass_contourA_fpath.toPlainText()
        control_group    = self.multiclass_control_group_fpath.toPlainText()
        control_group_id = self.multiclass_control_group_dspath.toPlainText()

        if  directory_path.strip() == "" or control_group.strip()=="":
            msg ="ERROR! Database or Control Group name not supplied"
            self.qm.about(self, "CRISP v0.2", msg)
            return         

        if os.path.isdir(directory_path):
            all_items = os.listdir(directory_path)                                                              # Get all items in the directory     
            folders = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]         # Filter out only the directories
            self.deepstack_multiclasses        = [ "/".join([directory_path,each_class]) for each_class in folders ] # set list for the multiAFRC proicessing
            self.deepstack_multiclass_control  =   "/".join([directory_path,control_group_id])                          # set control group for multi afrc processing

            xerox_path        = os.path.join(os.path.dirname(directory_path) , os.path.basename(directory_path) +"_deepstack")  # for copy purpise of main path
            xerox_cohort_path = os.path.join(os.path.dirname(self.multiclass_valid_cohort_pathname) , self.multiclass_val_cohort_source +"_deepstack")

            check_dir(xerox_path) if self.clone_deepstack_dataset else None                                            # creat folder if not exists
            check_dir(xerox_cohort_path) if self.clone_deepstack_dataset else None                                     # creat folder if not exists
        #===================================================================
        self.start_time =time.time()                                                                                    # initalize counter for profress bar
        #===================================================================                                            # Main loop to iterate over all classes vs Control
        for index, each_class in enumerate(self.deepstack_multiclasses):

            if os.path.basename(each_class) == os.path.basename(self.deepstack_multiclass_control):                     # skip teh deepstacking for Normal Class vs Normal Class
                continue
            else:
                self.logger("\n[%s] | Processing for : [%s]"%(time_stamp(), each_class), color =4 )
                self.multiclass_ds_pos_dspath.setText(os.path.basename(each_class))
                self.multiclass_ds_pos_fpath.setText(os.path.join(each_class, "rois_ds_info.data"))
                self.multiclass_control_group_dspath.setText(os.path.basename(self.deepstack_multiclass_control))
                self.multiclass_control_group_fpath.setText(self.deepstack_multiclass_control)

            #================================
            self.preview_multiclass_deepstacking_folder_list()       # show preview 
            self.multiclass_auto_compute_ssim_roi(show_msg = False)  # run the ROI computation
            #================================


            multiclass_stacked_array =self.multiclass_deep_stack                                                                                 # restore stack array values                                                                     
            if multiclass_stacked_array == None or len(multiclass_stacked_array) == 0:
                self.logger ("[%s] | Error! No stacking array detected. Use computue ROI with deepdtacking option enabled"%time_stamp() , color =5)       # check si teh stackinmg array is created
                return 
            
            #DeepStacking for CLass-A
            img_dir = os.path.dirname(self.multiclass_ds_pos_fpath.toPlainText())                                          # path of teh deep stacking image for disease group
            ClassA_imgList = get_source_images(img_dir)                                                                    # get list of all supported images in CLass A folder
            self.logger("\n[%s] | Building DeepStacked images for Class-A[%s]"%(time_stamp(),self.multiclass_ds_pos_dspath.toPlainText()))         
            check_dir(os.path.join(img_dir, "stacked_images")) 
    
            for raw_img in tqdm(ClassA_imgList):
                img =   stacked_img(cv2.imread(raw_img), stacking_array = multiclass_stacked_array)
                fnam =  os.path.join(img_dir,"stacked_images","stacked_" + os.path.basename(raw_img))
                cv2.imwrite(fnam, img)

                #=============
                if self.clone_deepstack_dataset:
                    check_dir(os.path.join(xerox_path, self.multiclass_ds_pos_dspath.toPlainText()))
                    fnam_xerox = os.path.join(xerox_path, self.multiclass_ds_pos_dspath.toPlainText(), "stacked_" + os.path.basename(raw_img))
                    cv2.imwrite(fnam_xerox, img)
                #============   
            
            self.logger("\n[%s] | Dataset written to: %s/stacked_images"%(time_stamp(), img_dir ))
            if self.clone_deepstack_dataset:
                self.logger("\n[%s] | Deepstack class %s dataset cloned to: %s/stacked_images"%(time_stamp(),self.multiclass_ds_pos_dspath.toPlainText(),fnam_xerox) )


            #==================================================== For the INFERENCE class if exists
            #DeepStacking for CLass-A for inference
            if self.multiclass_val_cohort_exists:

                cohort_img_dir = os.path.join(self.multiclass_valid_cohort_pathname, self.multiclass_ds_pos_dspath.toPlainText() )    # path of teh deep stacking image for disease group in inference cohort
                Cohort_imgList = get_source_images(cohort_img_dir)                                                                    # get list of all supported images in CLass A folder
                self.logger("\n[%s] | Building DeepStacked images for inference cohort class [%s]"%(time_stamp(),self.multiclass_ds_pos_dspath.toPlainText()))         
                check_dir(os.path.join(cohort_img_dir, "stacked_images")) 
    
                for raw_img in tqdm(Cohort_imgList):
                    img =   stacked_img(cv2.imread(raw_img), stacking_array = multiclass_stacked_array)                               # sue same arrray fop same class
                    fnam =  os.path.join(cohort_img_dir,"stacked_images","stacked_" + os.path.basename(raw_img))
                    cv2.imwrite(fnam, img)

                    #=============
                    if self.clone_deepstack_dataset:
                        check_dir(os.path.join(xerox_cohort_path, self.multiclass_ds_pos_dspath.toPlainText()))
                        fnam_xerox = os.path.join(xerox_cohort_path, self.multiclass_ds_pos_dspath.toPlainText(), "stacked_" + os.path.basename(raw_img))
                        cv2.imwrite(fnam_xerox, img)
                    #============   
            
                self.logger("\n[%s] | Dataset written to: %s/stacked_images"%(time_stamp(), img_dir ))
                if self.clone_deepstack_dataset:
                    self.logger("\n[%s] | Deepstack class %s dataset cloned to: %s/stacked_images"%(time_stamp(),self.multiclass_ds_pos_dspath.toPlainText(), fnam_xerox) )
                #============================================================================
 

            #Deep Stacking For Class B  (control group )                           
            img_dir =self.multiclass_control_group_fpath.toPlainText()    
            ClassB_imgList = get_source_images(img_dir)                                                                     # get list of all supported images in CLass B folder              
            
            self.logger("[%s] | Building DeepStacked images for Class-B [%s]"%(time_stamp(),self.multiclass_control_group_fpath.toPlainText()))    
            
            control_ds_image_path = os.path.join(img_dir, "stacked_images_" + self.multiclass_ds_pos_dspath.toPlainText()) 
            
            check_dir(control_ds_image_path)                                                                                # control will have deepstacks for each group pair
            

            for raw_img in tqdm(ClassB_imgList):
                img =  stacked_img(cv2.imread(raw_img), stacking_array = multiclass_stacked_array)
                fnam = os.path.join(control_ds_image_path, os.path.basename(raw_img))
                cv2.imwrite(fnam, img)
                #=============
                if self.clone_deepstack_dataset:
                    check_dir(os.path.join(xerox_path, self.multiclass_control_group_dspath.toPlainText()))
                    fnam_xerox = os.path.join(xerox_path, self.multiclass_control_group_dspath.toPlainText(), "stacked_" + os.path.basename(raw_img))
                    cv2.imwrite(fnam_xerox, img)
                #============  
            
            self.logger("\n[%s] | Dataset written to: %s/stacked_images"%(time_stamp(),img_dir) )

            if self.clone_deepstack_dataset:
                self.logger("\n[%s] | Deepstack CONTROL dataset cloned to: %s/stacked_images"%(time_stamp(),os.path.dirname(fnam_xerox)) , color = 5  )

    
            self.multiclass_store_ssim_pos()
            self.logger("[%s] | Deep Stacking ROIs stored successfully for: %s vs %s"%(time_stamp(), self.multiclass_ds_pos_dspath.toPlainText(),\
                                                                                                     self.multiclass_control_group_dspath.toPlainText()), color =4)

            #==== Update progress bar
            self.update_pbar(len(self.deepstack_multiclasses), index)
            #====


        self.logger("\n[%s] | Deepstacking dataset build completed for multiclass data "%time_stamp(), color =4 )


    def set_SSIM_value(self):

    	imgA,imgB =self.Contour_classA.copy(), self.Contour_classB.copy()                                                  # make copies as the .rect writes in image location

    	sel_index      = self.SSIM_list.currentRow()                                                                       # get current selected row    	
    	self.win_start = self.max_diff[sel_index ][1][0]                                                                   # for teh first time the o index items will be auto selected
    	self.win_end   = self.max_diff[sel_index ][1][1]

    	print("\n===================================")
    	print(Style.BRIGHT + Fore.YELLOW + "Currently selected index             : %d"%self.SSIM_list.currentRow()  )   
    	print(Style.BRIGHT + Fore.WHITE  + "Selected window location             :(%d - %d)"%(self.win_start , self.win_end))
    	print(Style.BRIGHT + Fore.WHITE  +"SSIM value                           :", self.max_diff[sel_index ][2])
    	print(Style.BRIGHT + Fore.WHITE  +"Search window size                   :", self.max_diff[sel_index ][3])  
    	print(Style.BRIGHT + Fore.WHITE  +"No. of scan size                     :", self.win_segment )
    	print(Style.BRIGHT + Fore.WHITE  +"Scan windows shift                   :", self.win_shift )
    	print(Style.BRIGHT + Fore.WHITE  +"Scan SSIM thresthold                 :", self.ssim_thresthold )

    	if self.preview_auto_roi :
    		cv2.rectangle(imgA, (self.win_start  ,1), ( self.win_end , imgA.shape[1]), (225, 128, 255), 2)
    		cv2.rectangle(imgB, (self.win_start  ,1), ( self.win_end , imgB.shape[1]), (225, 128, 255), 2)
    		imgA = cv2.copyMakeBorder(imgA,10,10,10,10,cv2.BORDER_CONSTANT,value= [0,0,0])     # vorder colour value = bloack i.ed [0,0,0]
    		imgB = cv2.copyMakeBorder(imgB,10,10,10,10,cv2.BORDER_CONSTANT,value= [0,0,0])
    		class_a, class_b = self.multiclass_ds_pos_dspath.toPlainText(), self.multiclass_control_group_dspath.toPlainText(),
    		hstack_title ="TOP: Class-A [%s] |Bottom: Class-B[%s] | mode: %s | SN. %d | Sim. score : %0.8f | win. position: %d-%d"%(class_a, class_b,self.ssim_mode, sel_index, self.max_diff[sel_index][2], self.win_start, self.win_end)
    		cv2.destroyAllWindows() 
    		cv2.imshow(hstack_title, cv2.vconcat((imgA, imgB)))
    		cv2.waitKey(0)
    		cv2.destroyAllWindows()  


    # MULTICLASS....      
    def set_multiclass_SSIM_value(self):

        imgA,imgB =self.multiclass_Contour_classA.copy(), self.multiclass_Contour_classB.copy()                                       # make copies as the .rect writes in image location

        sel_index      = self.multiclass_SSIM_list.currentRow()                                                                       # get current selected row       
        self.multiclass_win_start = self.multiclass_max_diff[sel_index ][1][0]                                                        # for teh first time the o index items will be auto selected

        self.multiclass_win_end   = self.multiclass_max_diff[sel_index ][1][1]
        print("\n===================================")
        print(Style.BRIGHT + Fore.YELLOW +"Currently selected index             : %d"%self.multiclass_SSIM_list.currentRow()  )   
        print(Style.BRIGHT + Fore.WHITE  +"Selected window location             :(%d - %d)"%(self.multiclass_win_start , self.multiclass_win_end))
        print(Style.BRIGHT + Fore.WHITE  +"SSIM value                           :", self.multiclass_max_diff[sel_index ][2])
        print(Style.BRIGHT + Fore.WHITE  +"Search window size                   :", self.multiclass_max_diff[sel_index ][3])  
        print(Style.BRIGHT + Fore.WHITE  +"No. of scan size                     :", self.multiclass_win_segment )
        print(Style.BRIGHT + Fore.WHITE  +"Scan windows shift                   :", self.multiclass_win_shift )
        print(Style.BRIGHT + Fore.WHITE  +"Scan SSIM thresthold                 :", self.multiclass_ssim_thresthold )
        
        if self.multiclass_preview_auto_roi:
            cv2.rectangle(imgA, (self.multiclass_win_start  ,1), ( self.multiclass_win_end , imgA.shape[1]), (225, 128, 255), 2)
            cv2.rectangle(imgB, (self.multiclass_win_start  ,1), ( self.multiclass_win_end , imgB.shape[1]), (225, 128, 255), 2)
            imgA = cv2.copyMakeBorder(imgA,10,10,10,10,cv2.BORDER_CONSTANT,value= [0,0,0])     # vorder colour value = bloack i.ed [0,0,0]
            imgB = cv2.copyMakeBorder(imgB,10,10,10,10,cv2.BORDER_CONSTANT,value= [0,0,0])
            hstack_title ="Class-A [TOP] Class-B [Bottom]| mode: %s | SN. %d | Similarity Score : %0.8f | window position: %d-%d"%(self.multiclass_ssim_mode, sel_index, self.multiclass_max_diff[sel_index][2], \
                                                                                                                                   self.multiclass_win_start, self.multiclass_win_end)
            cv2.destroyAllWindows() 
            cv2.imshow(hstack_title, cv2.vconcat((imgA, imgB)))
            cv2.waitKey(0) if self.multiclass_pause_each_afrc else  cv2.waitKey(2000) 
            cv2.destroyAllWindows()  


    def get_roi_from_coordinates(self):
        self.set_updated_gui_opts()        

        if not self.roi_from_cord_flag:
        	return

        # Warn if the the ROI is locked
        
        if self.roi_locked and self.roi_dims != None:
            self.logger("[%s] | Warning! ROI is locked with previous co-ordinates. Unlock ROI to continue..."%time_stamp(), color =5)
            if  self.qm.question(self,'CRISPII',"ROI is locked with previous co-ordinates. Clear previous ROI and continue?" , self.qm.Yes | self.qm.No) == self.qm.Yes:
                self.lock_roi_val_flag.setChecked(False)
                self.roi_locked = False
                pass
            else:
                self.logger("[%s]"%time_stamp() + " | Warning! The ROI value remains unchanged. ROI: (%d,%d,%d,%d) "%self.roi_dims , color = 5 )

                return
        #===========================================                  ROIs values
        y1 = self.sel_r2_start_time_slider.value()
        y2 = self.sel_r2_end_time_slider.value()
        x1 = self.sel_r1_start_time_slider.value()
        x2 = self.sel_r1_end_time_slider.value()
        #===========================================
        if (y2 < y1 ) or (x2 < x1):
        	self.logger("[%s] | Error! R1 and R2 End Co-ordinates must have high values than its Start co-ordinates. Input again!"%time_stamp(), color =5)
        	return

        self.roi_dims = (y1,y2,x1,x2)
        roi_image   = cv2.imread(self.roi_imagepath,cv2.IMREAD_COLOR)
        self.draw_roi_for_preview(img= roi_image.copy())

    def get_roi_from_template(self):
        self.set_updated_gui_opts()                                                                             # Hold the x1,y1,x2,y2 co-ordinates
        if self.roi_imagepath =="":
            self.logger("Select the template training image for setting roi first!")
            return                                                                                        

        if self.roi_locked and self.roi_dims != None:
            self.logger("[%s] | Warning! ROI is locked with previous co-ordinates. Unlock ROI to continue..."%time_stamp(), color =5)
            if  self.qm.question(self,'CRISPII',"ROI is locked with previous co-ordinates. Clear previous ROI and continue?" , self.qm.Yes | self.qm.No) == self.qm.Yes:
            	self.lock_roi_val_flag.setChecked(False)
            	self.roi_locked = False
            	pass
            else:
            	self.logger("[%s]"%time_stamp() + " | Warning! The ROI value remains unchanged. ROI: (%d,%d,%d,%d) "%self.roi_dims , color = 5 )

            	return

        roi_image        = cv2.imread(self.roi_imagepath,cv2.IMREAD_COLOR)                                           # read the original template image
        self.roi_dims    = get_rois_cord(cv2.imread(self.roi_imagepath))                                             # get the roi co-ordinates for cropping and store in roi dims

        if self.roi_dims == None:
            self.logger("[%s] | Warning! ROI HAS NOT been selected. Select ROI from image or manually supply ROI coordinates."%time_stamp(), color = 5)
            return
        self.draw_roi_for_preview(img= roi_image.copy())                                                                             # show roi preview 

    def draw_roi_for_preview(self, img):
        y1,y2,x1,x2 = self.roi_dims
        roi_image =img
        self.logger ("[%s] | Processed ROI co-ordinates: (%d, %d, %d, %d)"%(time_stamp(),y1,y2,x1,x2 ) )
        roi_image[y1:y2,x1:x2] = img_brightness(roi_image[y1:y2,x1:x2] )                                       # show the portion of the image brigntened
        roi_image = cv2.rectangle(roi_image, (x1,y1), (x2,y2), color = (155, 125, 0), thickness = 5)           # draw a box aroudn erectangle
        roi_image = image_resize(roi_image, width =700)                                                      # resize the image to fit the preview window
        self.roi_preview_frame.setPixmap(QPixmap.fromImage(self.displayImage(roi_image)))
    
    def set_roi_preview(self):   
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog 	
        fname, _ = QFileDialog.getOpenFileName(self,"Open source image file(.jpg) ",""," All Files (*.*);;Image Files (*.jpg, *.png, *.jpeg, *.tif)", options=options)
        if fname   == "":
        	self.logger("[%s] | Warning! the roi selection step was skipped"%time_stamp(), color =5)
        	return
        else:
        	self.roi_imagepath = fname
        	self.roi_template_image_path.setText(fname)
        	self.org_dim_rois_extract.setText(os.path.dirname(fname))                               #automatically set the pathname
        	#----
        	self.roi_ds_id.setText(os.path.basename(os.path.dirname(self.roi_template_image_path.toPlainText())) )   # set Fodlername as  ID for the single ROi datafile
        	#----
        	pass
        roi_image = cv2.imread(fname,cv2.IMREAD_COLOR)
        roi_image = image_resize(roi_image, width =700)
        self.roi_preview_frame.setPixmap(QPixmap.fromImage(self.displayImage(roi_image)))
        #========= Set limit for the manual co-ordinates selcxtion

        self.sel_r1_end_time_slider.setMaximum(cv2.imread(self.roi_imagepath).shape[1])              # R1 = width of image (max)
        self.sel_r2_end_time_slider.setMaximum(cv2.imread(self.roi_imagepath).shape[0] )             # R2 = height of image (max)
        self.sel_r1_start_time_slider.setMaximum( cv2.imread(self.roi_imagepath).shape[1])              # R1 = width of image (max)
        self.sel_r2_start_time_slider.setMaximum(cv2.imread(self.roi_imagepath).shape[0] )            # R2 = height of image (max)

    def show_cropped_roi_image(self):                                                               # check if the filr and ROI are selected
    	self.set_updated_gui_opts()
    	if self.roi_imagepath =="" or self.roi_dims == None:
    		self.logger("[INFO] Select the template training image for setting roi first!")
    		return
   				
    	roi_preview =  cv2.imread(self.roi_imagepath)
    	y1,y2,x1,x2 = self.roi_dims
    	roi_preview=  roi_preview[y1:y2,x1:x2]                                        # brighten the ROI section to make it clearly visible
    	cv2.imshow("Cropped ROI of Contour plot", roi_preview )    	                  # show image

    def show_org_roi_image(self):
        self.update_vars() # self.set_updated_gui_opts()
        if self.roi_imagepath =="":
        	self.logger("[INFO] Select the template training image for setting roi first!")
        	return
        cv2.imshow("Original Contour plot", cv2.imread(self.roi_imagepath)  )
    	#self.get_roi_from_coordinates()                                              # todo  remove

    def orgdim_rois_extract(self):
        self.set_updated_gui_opts()

        if self.roi_dims == None:
            self.qm.about(self,'CRISP',"Error! single ROI selection is empty. Select ROI from [GUI ROI selection] or Manually supply ROI coordinates first!")
            self.logger("[%s] | Warning! ROI HAS NOT been selected. Select ROI from image or manually supply ROI coordinates first."%time_stamp(), color = 5)
            return

        if self.qm.question(self,'CRISPII',"Extract selected ROIs with original dimensions for \n%s?"%self.orgdim_img_type , self.qm.Yes | self.qm.No) == self.qm.No:
            self.logger("[INFO] Original dimension ROIs extraction selection skipped....")
            return

        self.init_logger() 
        self.update_vars()
        self.run_mode = "Extract_ROIs"
        if self.apply_auto_roi_btn.isChecked() == True:
            self.logger("[%s] | Warning! the ROI: %d - %d (R1 lengthwise-wise) has been applied for all training images. R2 dimensions will remain unchanged!"%(self.win_start, self.win_end), color =5)
            self.use_computed_roi = True
       
        #===============Extracting ORGDIM ROIS fro IFAG
        self.logger("Source images:",self.raw_img_srcpath)                                    # 
        y1,y2,x1,x2 = self.roi_dims                                                           # get the roi dimensions
        self.logger("[%s] | Executing ROIs extraction mode..."%time_stamp())
        src_filelist = get_source_images(self.raw_img_srcpath)                                # function to get different format imge files
        self.logger("[%s] | Importing original source images..."%time_stamp())                # for original Normal mode and first import
        src_org_image_array = [cv2.imread(img_fname) for img_fname in tqdm(src_filelist) ]    # read the file into image array

        self.Images  = [ img_roi[y1:y2, x1:x2] for img_roi in src_org_image_array ]           # Extract ROIs and fill teh ROIs in arrays

        if self.orgdim_resize == True:
        	self.logger("\n[%s] | Resizing all original dimension image(s) to (h,w): (%d x %d)"%(time_stamp(), self.odim_height,self.odim_width))
        	self.Images = [cv2.resize(img_fname, (self.odim_width,self.odim_height))          # Resize all images
        	                                             for img_fname in tqdm(self.Images ) ]

        proc_path    = self.orgdim_roi_path + "/orgdim_" + os.path.basename(os.path.dirname(self.raw_file_path)) + \
                       "_imageset_" + self.orgdim_img_type.split("::")[0]  # folder path to store unresized images
        print("Saving preprocessed images....", self.Images[0].shape , " ROI:", self.roi_dims  )
        self.logger("\n[%s] | Processing path: %s"%(time_stamp(),proc_path) )

        #============================================
        # for Threshold mode
        if self.orgdim_img_type.count("Type_FXTH")== 1:
            self.logger("[%s] | Using fixed gradient thresholding (FXTH) mode..."%time_stamp()) 
            self.Images= get_threshold_images(self.Images, threshold = self.grad_threshold )  # 128

        # For adaptive gardient thesholding
        elif self.orgdim_img_type.count("Type_AGTH")== 1:
            self.logger("[%s] | Using Adaptive Gradient Threshold (AGTH) mode..."%time_stamp())

            if  self.adaptive_trendline == True:
                check_dir(os.path.join(self.orgdim_roi_path, "trendline_graph"))
                draw_trendline =   os.path.join(self.orgdim_roi_path, "trendline_graph" , "sample_trendline.jpg")
            else:
                draw_trendline = None

            self.Images= adaptive_gradient_threshold(self.Images, threshold_value=self.adaptive_threshold, output_mode=self.adaptive_color, 
                                                                  ksize_x=self.adaptive_kernel, ksize_y=self.adaptive_kernel, 
                                                                  sort_option= self.adaptive_sort_mode, 
                                                                  rndseed = self.adaptive_seed, 
                                                                  show_store_plot_fname =  draw_trendline)


        elif self.orgdim_img_type.count("Type_NRML") == 1 :
        	self.logger("[%s] | Saving in Normal image mode..."%time_stamp())          	
        	pass                                                                               # use the normal image as copied earlier
        #============================================
        self.store_processed_images(proc_path, fname = None, rgb_swap = False)                 # it will bydefault process all orgdim images in self.Image array
        #============================================================    
        self.Images =[]                                                                       # Must reset images
        #============================================================
        self.logger("[%s] | Saved to: %s"%(time_stamp(), proc_path ) )


    def preprocess_multiclass_images(self):
        if self.qm.question(self,'CRISPII',"Batch preprocess multiclass contour image source? " , self.qm.Yes | self.qm.No) == self.qm.No:
            return

        self.set_updated_gui_opts()
        self.run_mode = "Extract_multiclass_ROIs"
        # Get the directory path from the QLineEdit
        directory_path   = self.multiclass_source_datapath

        if  directory_path.strip() == "" :
            msg ="ERROR! source multiclass database path not supplied"
            self.qm.about(self, "CRISP v0.2", msg)
            return         

        if os.path.isdir(directory_path):
            all_items = os.listdir(directory_path)                                                              # Get all items in the directory     
            folders = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]         # Filter out only the directories
            self.multiclass_data     = [ "/".join([directory_path,each_class]) for each_class in folders ]      # set list for the multiclass dataset preprocessing paths containing each groups

            # print the dataset in terminal or console view
            self.logger("[%s] | Number of classes or groups"%time_stamp(), color =2)
            self.logger("--------------------------------------")
            for index,each_class in enumerate(self.multiclass_data):
                self.logger(f"{index} | {each_class}", color =3)
            self.logger("--------------------------------------")

            if self.apply_auto_roi_btn.isChecked() == True:
                self.logger("[%s] | Warning! the ROI: %d - %d (R1 lengthwise-wise) has been applied for all training images. R2 dimensions will remain unchanged!"%(self.win_start, self.win_end), color =5)
                self.use_computed_roi = True

            
            plt.ion() if not self.pause_each_preporcessing else None

            for index,each_class in enumerate(self.multiclass_data):
                self.logger("[%s] | Processing class: %s"%(time_stamp(), os.path.basename(each_class)))                       
       
                #===============Extracting ORGDIM ROIS fro IFAG
                self.logger("[%s] | Current source image class path:%s"%(time_stamp(),each_class), color =3 )                                   
                y1,y2,x1,x2 = self.roi_dims                                                           # get the roi dimensions
                self.logger("[%s] | Executing ROIs extraction mode..."%time_stamp())
                src_filelist = self.src_filelist = get_source_images(each_class)                 # get list of all fiels in current class

                self.logger("[%s] | Importing original source images..."%time_stamp())                # for original Normal mode and first import
                src_org_image_array = [cv2.imread(img_fname) for img_fname in tqdm(src_filelist) ]    # read the file into image array

                self.Images  = [ img_roi[y1:y2, x1:x2] for img_roi in src_org_image_array ]           # Extract ROIs and fill teh ROIs in arrays


                if self.orgdim_resize == True:
                    self.logger("\n[%s] | Resizing all original dimension image(s) to (h,w): (%d x %d)"%(time_stamp(), self.odim_height,self.odim_width))
                    self.Images = [cv2.resize(img_fname, (self.odim_width,self.odim_height))          # Resize all images
                                                         for img_fname in tqdm(self.Images ) ]

                proc_path    = each_class + "/orgdim_"   + os.path.basename(each_class) + \
                                                      "_imageset_" + self.orgdim_img_type[: self.orgdim_img_type.find("::")]  # folder path to store unresized images
                print("Saving pre-porcessed images....", self.Images[0].shape , " ROI:", self.roi_dims  )
                self.logger("\n[%s] | Processing path: %s"%(time_stamp(),proc_path) )

                #============================================
                # for Threshold mode
                if self.orgdim_img_type.count("Type_FXTH")== 1:
                    self.logger("[%s] | Using fixed gradient thresholding (FXTH)..."%time_stamp())                         
                    self.Images= get_threshold_images(self.Images, threshold = self.grad_threshold )  # 128

                # For adactive gardient thesholding
                elif self.orgdim_img_type.count("Type_AGTH")== 1:
                    self.logger("[%s] | Using Adaptive Gradient Threshold mode..."%time_stamp())

                    if  self.adaptive_trendline == True:
                        check_dir(os.path.join(each_class, "trendline_graph"))
                        draw_trendline =   os.path.join(each_class, "trendline_graph" , "sample_trendline.jpg")
                    else:
                        draw_trendline = None

                    self.Images= adaptive_gradient_threshold(self.Images, threshold_value=self.adaptive_threshold, output_mode=self.adaptive_color, 
                                                                  ksize_x=self.adaptive_kernel, ksize_y=self.adaptive_kernel, 
                                                                  sort_option= self.adaptive_sort_mode, 
                                                                  rndseed = self.adaptive_seed, 
                                                                  show_store_plot_fname =  draw_trendline)


                elif self.orgdim_img_type.count("Type_NRML") == 1:
                    self.logger("[%s] | Saving in normal image mode..."%time_stamp())           
                    pass                                                                               # use the normal image as copied earlier
                #============================================
                self.store_processed_images(proc_path, fname = self.src_filelist , rgb_swap = False)   # it will bydefault sore all orgdim images in self.Image assay
                #=============================================
                if self.clone_multiclass_dataset:
                    extract_suffix = self.orgdim_img_type.split("::")[0].strip("Type_") + "_"                                       # Plist at : ;and remove Type
    
                    clone_dirname     =  "cloned_" + extract_suffix + os.path.basename(self.multiclass_source_datapath)                                              # set the anme of main top source dataset with Type_Dataset                
                    clone_dirpathname =  os.path.join(os.path.dirname(self.multiclass_source_datapath), clone_dirname, os.path.basename(each_class) )    # sourc dir path, new source dir name, class name
                    print("Clone_dir pathanme: ", clone_dirpathname )

                    self.logger("[%s] | Cloning preprocess data to path: %s"%(time_stamp(), clone_dirpathname), color = 3)
                    check_dir(clone_dirpathname)                                                                                                         # make cole directory with same structure  
                    self.store_processed_images(clone_dirpathname , fname = self.src_filelist, rgb_swap = False) 
                
                #============================================================    
                self.Images =[]                                                                       # Must reset images
                #============================================================
                self.logger("[%s] | Saved to: %s"%(time_stamp(), proc_path ) )

            plt.ioff() if not self.pause_each_preporcessing else None                                 # remove auto image show at end

            if self.autoset_cloned_multiclass and self.clone_multiclass_dataset:                      # auto set the path for multi-class AFRC consequtively if clone & autosep both is selected
                self.logger("[%s] | Setting up cloned dataset path for multi-class AFRC..."%time_stamp(), color = 4)
                self.ds_src_dataset_location.setText(os.path.dirname(clone_dirpathname))
  

            self.logger("\n[%s] | A total of %d class(es) preprocessed"%(time_stamp(), len(self.multiclass_data)), color =4)





    def apply_auto_roi(self):
        if self.qm.question(self,'CRISPII',"Apply auto ROI?" , self.qm.Yes | self.qm.No) == self.qm.No:
            self.logger("\n[INFO] Auto ROI selection skipped....")
            return

        self.run_mode = "Extract_ROI"
        if self.apply_auto_roi_btn.isChecked() == True:
            self.logger("==========================================")
            self.logger("\n[%s] | Warning! the ROI: %d - %d (R1 lengthwise-wise) has been applied for all training images. R2 dimensions will remain unchanged!"%(time_stamp(), self.win_start, self.win_end), color =5)
            self.logger("\n[%s] | The preview image location in Load custom template will be overwritten by the location of CLass-A image"%time_stamp(), color =5)
            self.use_computed_roi = True
            self.roi_imagepath    = self.cimageA_path                                             # overwrite the preview mage template
            self.roi_dims = (0, self.Contour_classA.shape[0] , self.win_start, self.win_end)      # set the new roi_s locations
            self.logger("ROI DIMS : (%d, %d, %d, %d)"%self.roi_dims)
            self.draw_roi_for_preview(img = self.Contour_classA.copy() )                          # setup and refresh the preview of the new ROI

    def run_IAFG(self):
        if self.qm.question(self,"CRISP",
                                 "Apply AFRC?  Note: Each Class image will be automatically replaced by single IAFG representative image" , 
                                  self.qm.Yes | self.qm.No) == self.qm.No:
            return

        self.init_logger() 
        self.run_mode = "Extract_ROI"
        self.update_vars()                                 # update variables
        self.set_updated_gui_opts()                        # to updatet he recent local GUi options 

        check_dir("./temp")
        ClassA_dir = os.path.dirname(self.cimageA_path)    # get image path for Class-A
        ClassB_dir = os.path.dirname(self.cimageB_path)    # get image path for Class-B


        if self.iagf_resize_flag:                           # please mare sure all images are from same size ( from orgdim to thereafter)
            new_shape =  (self.iagf_new_width, self.iagf_new_height)
        else:
        	new_shape = None

        self.start_time = time.time()

        self.logger("[INFO] Calculating AFRC for Class-A")

        self.iafg_duration =  self.iafg_duration if self.auto_duration == True else  "auto"
        self.iafg_fps      =  self.iafg_fps      if self.auto_fps == True else       "auto" 

        src_video = img_to_vid(fpath = ClassA_dir, 
                               fout  = "./temp/cyclic_vid_ClassA.mp4", 
                               fps   = self.iafg_fps, 
                               duration     = self.iafg_duration,
                               sharpen_flag = self.iafg_sharpen,
                               new_dim      = new_shape,
                               agu_dilate   = self.iafr_dilation,
                               agu_erode    = self.iafr_erosion,
                               agu_noised   = self.iafr_noise,
                               agu_denoised = self.iafr_denoise )

        ifag_img_A = avg_weight(fpath = src_video, win_title = "CLASS-A", show_preview = self.iafg_preview)

        check_dir(ClassA_dir +"/afrc_image/")
        cv2.imwrite(ClassA_dir +"/afrc_image/afrc_image.jpg",ifag_img_A)
        self.cimageA_path = ClassA_dir +"/afrc_image/afrc_image.jpg"

        self.contourA_dspath.setText(ClassA_dir    + "/afrc_image/afrc_image.jpg")    if self.autoset_iafg_path.isChecked() else None
        self.contourA_dspath_ds.setText(ClassA_dir )    if self.autoset_iafg_path.isChecked() else None        

        self.update_pbar(2,0)                                      # 50% comeplted

        self.logger("[INFO] Calculating AFRC for Class-B")
        src_video = img_to_vid(fpath = ClassB_dir, 
                               fout  = "./temp/cyclic_vid_ClassB.mp4", 
                               fps   = self.iafg_fps,
                               duration     = self.iafg_duration,
                               sharpen_flag = self.iafg_sharpen,
                               new_dim      = new_shape,
                               agu_dilate   = self.iafr_dilation,
                               agu_erode    = self.iafr_erosion,
                               agu_noised   = self.iafr_noise,
                               agu_denoised = self.iafr_denoise )

        
        ifag_img_B = avg_weight(fpath = src_video, win_title = "CLASS-B", show_preview = self.iafg_preview)

        
        check_dir(ClassB_dir +"/afrc_image/") 
        cv2.imwrite(ClassB_dir +"/afrc_image/afrc_image.jpg",ifag_img_B)
        self.cimageB_path = ClassB_dir +"/afrc_image/afrc_image.jpg"

        self.contourB_dspath.setText(ClassB_dir +"/afrc_image/afrc_image.jpg")    if self.autoset_iafg_path.isChecked() else None
        self.contourB_dspath_ds.setText(ClassB_dir )                              if self.autoset_iafg_path.isChecked() else None   

        self.update_pbar(2,1)  # 100% complete

        #====Display results
        ifag_img_A= cv2.copyMakeBorder(ifag_img_A,5,5,5,5,cv2.BORDER_CONSTANT,value= [0,0,0])
        ifag_img_B= cv2.copyMakeBorder(ifag_img_B,5,5,5,5,cv2.BORDER_CONSTANT,value= [0,0,0])
        class_a,class_b =  os.path.basename(ClassA_dir) , os.path.basename(ClassB_dir) 
        cv2.imshow(f"AFRC equivalent image | TOP: Class-A [{class_a}] | BOTTOM: Class-B [{class_b}]",  cv2.vconcat( ( scaled_resize(ifag_img_A), scaled_resize(ifag_img_B) )) )
        cv2.waitKey(0) 

    #===========================================================================

    def run_IAFG_multiclass(self):
        if self.qm.question(self,"CRISP-II",
                                 "Apply multi-class AFRC?  Note: Each Class contour will yield single IAFG representative image" , 
                                  self.qm.Yes | self.qm.No) == self.qm.No:
            return

        self.init_logger() 
        self.run_mode = "Extract_ROI"
        self.update_vars()                                 # update variables
        self.set_updated_gui_opts()                        # to updatet he recent local GUi options 
        check_dir("./temp")                                # create temp dir is not esxists        
        self.preview_multiclass_folder_list()              # auto run preview to set variabgles  :  self.multiclass_afrc_path and
        total_groups = len(self.multiclass_afrc_path)      # number of classes
        
        self.start_time = time.time()
        # Processing in batch
        for index, each_class in enumerate(self.multiclass_afrc_path):       # Loop through all classes

            ClassA_dir = each_class                        # get image path for each class except Normal
            class_a = os.path.basename(each_class)

            if self.iagf_resize_flag:                      # please mare sure all images are from same size ( from orgdim to thereafter)
                new_shape =  (self.multiclass_iagf_new_width, self.multiclass_iagf_new_height)
            else:
                new_shape = None

            self.logger("\n[%s] | Computing multi-class AFRC for Class-A [%s]"%(class_a, time_stamp()) , color = 4)

            self.iafg_duration =  self.iafg_duration if self.auto_duration == True else  "auto"
            self.iafg_fps      =  self.iafg_fps      if self.auto_fps == True else       "auto" 

            src_video = img_to_vid(fpath = ClassA_dir, 
                                fout  = "./temp/cyclic_vid_ClassA.mp4", 
                                fps   = self.multiclass_iafg_fps, 
                                duration     = self.multiclass_iafg_duration,
                                sharpen_flag = self.multiclass_iafg_sharpen,
                                new_dim      = new_shape,
                                agu_dilate   = self.multiclass_iafr_dilation,
                                agu_erode    = self.multiclass_iafr_erosion,
                                agu_noised   = self.multiclass_iafr_noise,
                                agu_denoised = self.multiclass_iafr_denoise )

            ifag_img_A = avg_weight(fpath = src_video, win_title = "CLASS-A", show_preview = self.multiclass_iafg_preview)

            check_dir(ClassA_dir +"/afrc_image/")
            cv2.imwrite(ClassA_dir +"/afrc_image/afrc_image.jpg",ifag_img_A)
            self.cimageA_path = ClassA_dir +"/afrc_image/afrc_image.jpg"


            #====Display results
            ifag_img_A= cv2.copyMakeBorder(ifag_img_A,5,5,5,5,cv2.BORDER_CONSTANT,value= [0,0,0])
            cv2.imshow(f"AFRC image for: Class-A [{class_a}]",  scaled_resize(ifag_img_A))
            if self.pause_each_afrc == True:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                cv2.waitKey(2000)
                cv2.destroyAllWindows()

            self.update_pbar(total_groups, index)


        self.logger("\n====================================================================================", color =4)
        self.logger("\n[%s] | AFRC construction completed for %d classes"%(time_stamp(), len(self.multiclass_afrc_path)), color = 4)


    #===========================================================================

    def loadImage(self,fname= "./logo.jpg"):
        self.image=cv2.imread(fname,cv2.IMREAD_COLOR)
        self.image = image_resize(self.image, height = 320)
        try:
        	self.trainOpenImg.setPixmap(QPixmap.fromImage(self.displayImage(self.image)))
        	self.trainOpenImg.setAlignment(QtCore.Qt.AlignCenter)
        except:
        	self.logger("[%s] | Preview update skipped due to potential memory error.."%time_stamp() ,color = 5)

    def displayImage(self, img):                                                     # Returns a qimage, must eb a colour image
        qformat =QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2] ==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888
            img = QtGui.QImage(img.data,
                img.shape[1],
                img.shape[0], 
                img.strides[0], qformat)
            img = img.rgbSwapped()
        return img

    def make_plot(self):
        self.plot_stats.removeItem(self.gfx1)     
        self.datos = pg.ScatterPlotItem()
        self.gfx1  = self.plot_stats.addPlot(title='loss f(disc.)')
        self.gfx1.setLabel('left','Log10(loss)')
        self.gfx1.setLabel('bottom','epoch(s)')
        self.datos = self.gfx1.plot(pen='y')
        self.datos.setData(self.DRloss_list) #(random.sample(range(100), 50))
        self.gfx1.enableAutoRange(x=True,y=True)
  
        self.plot_stats.removeItem(self.gfx2) 
        self.datos = pg.ScatterPlotItem()
        self.gfx2   = self.plot_stats.addPlot(title='loss f(gen.)')
        self.gfx2.setLabel('left','loss')
        self.gfx2.setLabel('bottom','epoch(s)')
        self.datos = self.gfx2.plot(pen='y')
        self.datos.setData(self.G_loss_list) 
        self.gfx2.enableAutoRange(x=True,y=True)

        self.cur_model_log.insertPlainText("\nnet iter. %d | dis. loss: %0.5f | gen. loss: %0.5f "%(self.net_gan_iter, self.DRloss_val, self.G_loss_val) )
        self.cur_model_log.setText("\nnet iter. %d | dis. loss: %0.5f | gen. loss: %0.5f "%(self.net_gan_iter, self.DRloss_val, self.G_loss_val) )

        if self.Eval_FID_flag and len(self.FID_list) > 0:                               # the FID_list NULL with creat error so need atleast 1 value if EVAL_FID
        	self.plot_stats.removeItem(self.gfx3) 
        	self.datos  = pg.ScatterPlotItem()
        	self.gfx3   = self.plot_stats.addPlot(title='FID')
        	self.gfx3.setLabel('left','FID scores')
        	self.gfx3.setLabel('bottom','epoch(s)')
        	self.datos = self.gfx3.plot(pen='y')
        	self.datos.setData([x[0] for x in  self.FID_list], [x[1] for x in self.FID_list] ) 
        	self.gfx3.enableAutoRange(x=True,y=True)

        #===========Blur loss
        if len(self.blur_scr_lst) > 0:
        	try:
        		self.plot_noise.removeItem(self.gfx_noise)
        	except:
        		pass

        	self.datos  = pg.ScatterPlotItem()
        	self.gfx_noise   = self.plot_noise.addPlot(title='Blur fac(gen.)')
        	self.gfx_noise.setLabel('left','Blurness ')
        	self.gfx_noise.setLabel('bottom','epoch(s)')
        	self.datos = self.gfx_noise.plot(pen='y')
        	self.datos.setData([ x[0] for x in self.avg_blr_history ], [ x[1] for x in self.avg_blr_history ] ) 
        	self.gfx_noise.enableAutoRange(x=True,y=True) 
        #==================                                                                         # Show the swarm plots for the generator values 
        if True:
        	self.make_swarm_plot()       


    def make_swarm_plot(self):                                                                    # swarmplt for the blur fac 
        try:
        	self.swarm_plot.removeItem(self.gfx_swarm_plot)                                       # update the graph for the swarm plots with error handeling 
        except:
        	pass 

        if self.blur_scr_lst == []:
        	return   
        
        self.value_n =len(self.blur_scr_lst)                                                      # holds the blurred scores for the sampleing batch

        self.value_n   = 10 if self.value_n > 10 else  self.value_n                               # show only of last 100 in interval of 10 epoches
        cur_blur_scores = self.blur_scr_lst[-10:] if self.value_n >10 else  self.blur_scr_lst     # showe only for last 100 in internal 10 epoches

        n_item_score=[]                                                                           # holds th item scores 

        for epoch in cur_blur_scores:
            n_item_score.append([j for j in epoch])
        data= np.array(n_item_score)

        self.gfx_swarm_plot   = self.swarm_plot.addPlot(title= 'qScore batch logs')
        self.gfx_swarm_plot.showGrid(x = True, y = True) 
        '''    
        if True:                                                                                     #=== add bar plot (on top if any)
            bar = pg.BarGraphItem(x=range(self.value_n), height=data.mean(axis=1), width= 0.1, brush=0.3)
            self.gfx_swarm_plot.addItem(bar)
        '''
        #==== add scatter plots (on top if any)
        if True:
            for i in range(self.value_n):
                xvals = pg.pseudoScatter(data[i], spacing=0.4, bidir=True) * 0.2                      # Given a list of x-values, construct a set of y-values such that an x,y scatter-plot will not have
                self.gfx_swarm_plot.plot(x=xvals+i, y=data[i], pen=None, symbol='o', symbolBrush=pg.intColor(i,6,maxValue=128))
       #=====add error bar on top (if any)
        if True:                                                                 # Make error bars if true
            err = pg.ErrorBarItem(x=np.arange(self.value_n), y=data.mean(axis=1),\
                  height=data.std(axis=1), beam=0.2, pen={'color':'w', 'width':1})
            self.gfx_swarm_plot.addItem(err)

        #======= print epoch
        self.gfx_swarm_plot.setLabel('left'  , 'blur qScores')
        self.gfx_swarm_plot.setLabel('bottom', 'batch values @ last iter. [n= %d]'%self.value_n)



    def image_effect_agument(self, cv_img):                                                 # this modules addes image agumentation efefcts

            if self.agument_dilated_img:
                kernel = (self.img_dilate_kernel,self.img_dilate_kernel)                    # imagew dilation
                agu_img = np.asarray(img_dilate(cv_img, kernel, self.img_dilate_iter),  dtype='float32')
                self.Images.append(agu_img)
                self.count_agumented +=1
                                                  
            if self.agument_eroded_img:                                                     # Image erosion 
                kernel = (self.img_dilate_kernel,self.img_erosion_iter)
                agu_img = np.asarray(img_erode(cv_img, kernel, self.img_dilate_iter),  dtype='float32')
                self.Images.append(agu_img) 
                self.count_agumented +=1

            if self.rnd_img_ehnance:                                                        # image contrast
                agu_img = np.asarray(img_contrast(cv_img), dtype='float32')
                self.Images.append(agu_img) 
                self.count_agumented +=1

            if self.rnd_img_brightness:                                                     # image brightness
                agu_img = np.asarray(img_brightness(cv_img), dtype='float32')
                self.Images.append(agu_img)
                self.count_agumented +=1

            if self.rnd_img_contrast:
                agu_img = np.asarray(img_contrast(cv_img), dtype='float32')
                self.Images.append(agu_img)
                self.count_agumented +=1

            if self.agument_noise and self.agument_gaussian:
                agu_img = np.asarray(agument_noise(cv_img, "gaussian", 5, self.noise_intenisty), dtype='float32')
                self.Images.append(agu_img)
                self.count_agumented +=1

            if self.agument_noise and self.agument_random :
                agu_img = np.asarray(agument_noise(cv_img, "random", 5, self.noise_intenisty), dtype='float32')
                self.Images.append(agu_img)
                self.count_agumented +=1

            if  self.agument_distortion == True:                                                                          # agument distortion
                agu_img = np.asarray(distorted_img(cv_img,  self.distortion_fac, output_dim = self.output_shape_value), dtype='float32')
                self.Images.append(agu_img)
                self.count_agumented +=1

            if self.hflip_flag == True:
                agu_img = np.asarray(rnd_flip(cv_img, flip_type= 1), dtype='float32')                                      # must be done on cv2 image
                self.Images.append(agu_img)                                                                                # RGB --> BGR  very important
                self.count_agumented +=1

            if self.agument_sharpened:
                agu_img = np.asarray(rnd_sharpen(cv_img, kernel_range = self.sharpen_val), dtype='float32')                # must be done on cv2 image
                self.Images.append(agu_img)
                self.count_agumented +=1

            if self.agument_blurred:
                agu_img = np.asarray(rnd_blurred(cv_img, kernel_range = self.blurred_val), dtype='float32')                # must be done on cv2 image
                self.Images.append(agu_img)
                self.count_agumented +=1

            if self.agument_denoised:                                                                                      # agument denoised images
                agu_img = np.asarray(rnd_denoise(cv_img, kernel_range = self.denois_val), dtype='float32')
                self.Images.append(agu_img)   
                self.count_agumented +=1


    def import_imgs(self):

        file_index =0                                                                                 # iniciate file counter
        src_filelist =[]                                                                              # iniciate the list of files
        src_org_image_array =[]                                                                       # teharray which hold the original images in to cv2 imread format
        src_image_array     =[]                                                                       # holds the image array for the mode-processed images in cv2 imread format
        
        src_filelist = get_source_images(self.source_image_path)                                      # function to get different format imge files
        # for original Normal mode and first import
        tqdm_info=("[%s] | Importing original source images "%time_stamp())
        src_org_image_array = [cv2.imread(img_fname) for img_fname in tqdm(src_filelist, desc=tqdm_info, ncols =100) ]            # read the original images into array from path
        
        #====================== this block corrspondes to use of manual ROI selection (first selected by user)
        if self.extract_using_roi:                                                                    # very important. extracts the data based on ROI and resizes them to 256 x 256 for furtehr processing
        	y1,y2,x1,x2 = self.roi_dims                                                                    # get the roi dimensions        	
        	self.logger("[%s] | Executing ROIs extraction mode..."%time_stamp())
        	src_org_image_array = [ img_roi[y1:y2, x1:x2] for img_roi in  tqdm(src_org_image_array) ] # make the original image to roi dim, resize and replace with the org array 
        	self.logger("[%s] | Resizing ROIs to fit on model"%time_stamp())
        	temp_array = [ cv2.resize(img_roi, (self.output_shape_value, self.output_shape_value), interpolation = cv2.INTER_AREA)             # resizing must be done only after cropping to preserve iformation
        	                                                       for img_roi in tqdm(src_org_image_array) ]
        	src_org_image_array = temp_array                                                          # assigne to original image array after processing
        	del temp_array                                                                            # Free up memory  as it can be alot of files 
        	src_image_array = src_org_image_array                                                     # for now src_image_array and src_org_image array is copied as clone
        
        elif self.use_computed_roi:                                                                   # for the auto-computed roi        
            self.logger("[%s] | Executing auto-ROI extraction mode..."%time_stamp())                                         
            src_org_image_array = [ img_roi[:, self.win_start: self.win_end ] for img_roi in  tqdm(src_org_image_array) ]

            temp_array = [ cv2.resize(img_roi, (self.output_shape_value, self.output_shape_value), interpolation = cv2.INTER_AREA)             # resizing must be done only after cropping to preserve iformation
                                                                 for img_roi in tqdm(src_org_image_array) ]
            src_org_image_array = temp_array 
            del temp_array                                                                            # free up these memory
            src_image_array = src_org_image_array      
        else:
        	tqdm_info = ("[%s] | Setting image(s) for model input "%time_stamp()) 
        	temp_array = [ cv2.resize(img_roi, (self.output_shape_value, self.output_shape_value), interpolation = cv2.INTER_AREA)             # resizing must be done only after cropping to preserve iformation
        	                                                       for img_roi in tqdm(src_org_image_array, desc =tqdm_info, ncols =100) ]
        	src_org_image_array = temp_array                                                                                       # assigne to original image array after processing
        	del temp_array                                                                                                         # Free up memory  as it can be alot of files 
        	src_image_array = src_org_image_array        
        #==============================================================================================

        # for threshold mode (experimental)
        if self.gan_train_type.count("Type_FXTH")== 1:                                    
            src_image_array = get_threshold_images(src_org_image_array, threshold = self.grad_threshold )  # FIrst,  append the HED images only with default values if selected 
            if self.randomize_grad:
                self.logger("[%s] | Agumenting Threshold Image..."%time_stamp())           # agument the HED images with randomized  mean values and get final list. Hereby, the list size will double
                src_image_array = src_image_array + get_threshold_images( src_org_image_array, self.grad_threshold )   

        
        del src_org_image_array                                                           # At this point, free up memory by deleting the original arary and keep only processed array (i.e. src_image_array)
        	                                                                              # Since the HNED corks on NN moduels, the loading/conversion is done at once for all images
        for each_img_in_array in  src_image_array :                                       # Iterate through the mode-processed  src_image_array 
            cv_img = each_img_in_array                                                    # Assign the singleimage in mode-porcessed arary to the cv_img variabel for furtehr agumentation & processing 
            cv_img = cv_img[:, :, ::-1]                                                   # RGB --> BGR  very important for cv2 to numpy image transition and normalize to 0.0 - 1.0
            file_index +=1                                                                # increment the index count
            tmp_org = np.asarray(cv_img)                                                  # convert cv2 to numpy array
            self.Images.append(tmp_org)                                                   # add image 

            self.image_effect_agument(cv_img)                                             # this function will agument lots of effect images to self.Images

                                                                                          # make dirs name : "image_datta_Type_MODE" if not present
        if self.store_processed_img and self.run_mode == "train_gan":                     # make only when selected
            proc_path =   self.processed_img_path + "/ID_"+ self.model_id +"_"  \
                          + self.gan_train_type[ : self.gan_train_type.find("::")]
            self.store_processed_images(proc_path) if self.store_processed_img  else None # store the entire processed images if requested
            self.logger("[%s] | Total of %d agumented image(s) were added"%(time_stamp(),self.count_agumented))
        #=============================================================================================================     # image agumentation/importing block finished here           

    def store_processed_images(self,proc_path , fname = None ,rgb_swap = True):       # module to store gan processed source images (org_cropped + agumented   	
    	check_dir(proc_path)                                                          # make dirs if not exists
    	tqdm_info = "[%s] | Storing all processed source images  :"%time_stamp()
                    

    	for index, each_image in enumerate(tqdm(self.Images ,desc =tqdm_info, ncols =100)):
    		each_image = each_image[:, :, ::-1]  if rgb_swap == True else each_image                               # RGB --> BGR  very important for cv2 to numpy image transition and normalize to 0.0 - 1.0

    		if isinstance(fname, list):                                                                                                                                                           
    			cur_fname, _ = get_filename_and_extension(self.src_filelist[index])                                # return single file name or original filename
    		else:                                                                                                                                                           
    			cur_fname = str(fname)                                                                             # return single file name or original filename


    		if self.orgdim_format == "jpg":
    			cv2.imwrite(proc_path +"/"+ cur_fname + "_processed"+ str(index) + "."+ self.orgdim_format, each_image , [int(cv2.IMWRITE_JPEG_QUALITY), 99])   # set JPG with highest quality
    		else:
    			cv2.imwrite(proc_path +"/"+ cur_fname + "_processed"+ str(index) + "."+ self.orgdim_format, each_image )                                        # for PNG save direct

    	if not (self.run_mode == "synthesis"):                                          # omly show messages during initial image prcessing for training
    		self.logger("\n[%s] | %d agumented image(s) written successfully"%(time_stamp(), index + 1) )

    def update_pbar(self,tot_index, cur_index):
        ETA=(tot_index - cur_index) * (time.time()-self.start_time)/(2*60)
        itm_per_sec =  2/(0.0001 + time.time()-self.start_time)                        # rough ETA where 0.0001 is epselon for time.time()
        self.start_time = time.time()
        self.progressBar.setValue(round(100 * (cur_index +1 ) /tot_index) )
        self.progress_status.setText("Status: %s network | ETA(min): %0.5f  | ~(epoch, iter., items)/s : %0.5f "%(self.run_mode, ETA, itm_per_sec) ) 

    #==================== GANTraining block

    def set_gan_plots(self):
        #=======================Inititate make plot block
        self.gfx1 = self.plot_stats.addPlot()
        self.gfx2 = self.plot_stats.addPlot()
        if self.Eval_FID_flag:
        	self.gfx3   = self.plot_stats.addPlot()                   
        self.gfx_noise  = self.plot_noise.addPlot()       
        self.make_plot()                                        
        #=================================

    def construct_dis(self):                                                        # core discriminator for the GANs
        # Discriminator
            x_in = Input(shape=(self.img_dim, self.img_dim, 3))
            x = x_in
            for i in range(self.num_layers + 1):
                num_channels = self.max_num_channels // 2**(self.num_layers - i)
                x = Conv2D(num_channels, (5, 5), strides=(2, 2),
                                                use_bias=False,
                                                padding='same')(x)
                if i > 0:
                    x = BatchNormalization()(x)
                x = LeakyReLU(alpha=self.dis_lrelu_alpha)(x)  if self.dis_lrlu_flag else Activation('relu')(x)  # original LeakyReLU(0.2)(x)            

            x = Flatten()(x)
            x = Dense(1, use_bias=False)(x)      
            
            d_model = Model(x_in, x)
            d_model.summary() if self.show_model_summary == True else None          # show summary
            
            return d_model

    def construct_gen(self):
        # Generator
        z_in = Input(shape=(self.z_dim, ))
        z = z_in
        z = Dense(self.f_size**2 * self.max_num_channels)(z)
        z = BatchNormalization()(z)                                                                      # experimental
        z = LeakyReLU(alpha=self.gen_lrelu_alpha)(z)  if self.gen_lrlu_flag else  Activation('relu')(z)  #==========#leakyRelu added here https://www.machinecurve.com/index.php/2019/11/12/using-leaky-relu-with-keras/
        z = Reshape((self.f_size, self.f_size, self.max_num_channels))(z)
        for i in range(self.num_layers):
            self.num_channels = self.max_num_channels // 2**(i + 1)
            z = Conv2DTranspose(self.num_channels, (5, 5), strides=(2, 2), padding='same')(z)
            z = BatchNormalization()(z)
            #===================================== Added new layers my  me
            z=  Dense(self.num_channels, activation=  (LeakyReLU(alpha=self.gen_lrelu_alpha) if self.gen_lrlu_flag  else 'relu')   )(z)
            z = BatchNormalization()(z)
            #=====================================
            z = LeakyReLU(alpha=self.gen_lrelu_alpha)(z)  if self.gen_lrlu_flag  else  Activation('relu')(z)
        z = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(z)
        z = Activation('tanh')(z)
        g_model = Model(z_in, z)
        g_model.summary()   if self.show_model_summary == True else None   
        return g_model
        

    def integrated_model(self):                                      
        # Integrated model (training discriminator)
        x_in = Input(shape=(self.img_dim, self.img_dim, 3))
        z_in = Input(shape=(self.z_dim, ))
        g_model = self.construct_gen()
        g_model.trainable = False

        x_real = x_in
        x_fake = g_model(z_in)

        d_model      = self.construct_dis()
        x_real_score = d_model(x_real)
        x_fake_score = d_model(x_fake)

        d_train_model = Model(  [x_in, z_in], [x_real_score, x_fake_score]  )

        d_loss = x_real_score - x_fake_score
        d_loss = d_loss[:, 0]
        d_norm = 10 * K.mean(K.abs(x_real - x_fake), axis=[1, 2, 3])
        d_loss = K.mean(- d_loss + 0.5 * d_loss**2 / d_norm)

        d_train_model.add_loss(d_loss)

        #===select type of optimizer used
        d_train_model.compile(optimizer=Adam(self.gan_lr , beta_1= 0.5, clipvalue=0.5))  if self.gmodel_optimizer == "Adam" else None  # Adadelta   #default Adam(2e-4, 0.5)                                        # Learning rate and momentum can be fixed here....
        d_train_model.compile(optimizer=SGD (self.gan_lr, clipvalue=0.5))      if self.gmodel_optimizer == "SGD" else None
        d_train_model.compile(optimizer=RMSprop(self.gan_lr))  if self.gmodel_optimizer  == "RMSprop" else None
        d_train_model.compile(optimizer=Adamax(self.gan_lr))   if self.gmodel_optimizer  == "Adamax"  else None
        d_train_model.compile(optimizer=Nadam(self.gan_lr))    if self.gmodel_optimizer  == "Nadam"   else None

        # Integrated model (training generator)
        g_model.trainable = True
        d_model.trainable = False

        x_real = x_in
        x_fake = g_model(z_in)

        x_real_score = d_model(x_real)
        x_fake_score = d_model(x_fake)

        g_train_model = Model(  [x_in, z_in],   [x_real_score, x_fake_score]  )

        g_loss = K.mean(x_real_score - x_fake_score)

        g_train_model.add_loss(g_loss)
        g_train_model.compile(optimizer=Adam(2e-4, 0.5))

        # Check model structure for discriminator and geneator
        d_train_model.summary()  if self.show_model_summary == True else None   
        g_train_model.summary()  if self.show_model_summary == True else None   

        return (d_train_model, g_train_model, g_model)


    def sample(self, path, g_model,  n=9, z_samples=None , gan_fname = "" ):         
                                                                         # gan_fname is associated with the naming of synthetic images
        figure = np.zeros((self.img_dim * n, self.img_dim * n, 3))
        if z_samples is None:
            z_samples = np.random.randn(n**2, self.z_dim)
        for i in range(n):
            for j in range(n):
                z_sample = z_samples[[i * n + j]]

                if self.run_mode == "synthesis":
                    x_sample = self.g_model_synthesis.predict(z_sample)  # to preserve memory use and make fast same model during syntehsis, else its very slow
                else:
                    x_sample = g_model.predict(z_sample)

                digit = x_sample[0]
                figure[ i * self.img_dim:(i + 1) * self.img_dim,
                        j * self.img_dim:(j + 1) * self.img_dim ] = digit
        figure = (figure + 1) / 2 * 255
        figure = np.round(figure, 0).astype(int)                         # warning comes here of image lossy due to flot32 to int conversion, but its ok
        imageio.imwrite(path, figure)                                    # Sampling function

        if self.gan_agument_flag and self.run_mode == "synthesis" :      # should read the image in cv2 format, so better just to read from the output file
            self.Images = []                                             #  reset images (we do one image per time as batch operation can overrun memory)
            cv_img =cv2.imread(path)
            cv_img = cv_img[:, :, ::-1]                                  # RGB --> BGR  very important for cv2 to numpy image transition and normalize to 0.0 - 1.0
            self.image_effect_agument(cv_img)                            #  generate agumented images for the current synthetic images  in self.Images[]   
            proc_path =  self.gen_imgs_path +  "/" + "_".join([self.model_id, str(self.synth_model_dim), self.gan_timestamp]) + "_agumented"   
            self.store_processed_images( proc_path , fname = gan_fname) # write all agument images for the given synthetic images



    def blur_sampling(self, g_model = None, n=3):
        blur_vals =[]
        for _ in range(n):
            z_samples = np.random.randn(1, self.z_dim)
            z_sample  = z_samples[[0]]
            x_sample  = g_model.predict(z_sample)[0]
            figure    = matplot_to_cv2(x_sample)
            blur_vals.append(blur_fac(figure))
            #cv2.imshow("C.R.I.S.P:Preview", figure)
            #cv2.waitKey(0)
        self.blur_scr_lst.append(blur_vals)
        self.avg_blr_history.append( ( self.net_gan_iter, sum(blur_vals)/len(blur_vals) ) )        # add the last value to the average blur list
    
        return  blur_vals


    def evaluate(self, g_model, eval_imgs = 4):                                                     # eval images are the n for n x n images (column or row)
        im_no = random.randint(0, len(self.Images) - 1)
        im1   = self.Images[im_no]
        z_samples = np.random.randn(eval_imgs**2, self.z_dim)
        img_array = []
        #plt.close() 

        for i in range(eval_imgs):
            for j in range(eval_imgs):
                z_sample = z_samples[[i * eval_imgs + j]]
                x_sample = g_model.predict(z_sample)
                #plt.imshow(x_sample[0],  interpolation='nearest')
                #plt.show()
                img_array.append(x_sample[0])                                     

        grid_imgs = int(eval_imgs**0.5)   
        #=========================
        fig, axarr = plt.subplots(nrows=grid_imgs , ncols=grid_imgs)
        plt.title("[%s] | CRISP Preview at:%d epoch"%(time_stamp(), self.cur_epoch))
        prv_img_count =0
        row_count = 0

        #======================                                        
        for row in axarr:
            col_count =0
            for col in row:
                plt.title("Figure" + str(prv_img_count) )
                axarr[row_count,col_count].axis('off')
                axarr[row_count,col_count].set_title('Simulation-' + str(prv_img_count)) if eval_imgs <=9 else \
                                           axarr[row_count,col_count].set_title("sim"+str(prv_img_count))                    
                #=============
                if prv_img_count == 0:
                    col.imshow(im1.astype('uint8'))
                    axarr[row_count,col_count].set_title('Ground Truth')
                else:
                    col.imshow((img_array[prv_img_count] * 255).astype('uint8'))              
                prv_img_count +=1                                                  
                col_count +=1                                                     
                #=============

            row_count +=1
   
        fig.text(0.5, 0.04, '1D Retension time -->', ha='center', va='center')
        fig.text(0.06, 0.5, '2D Retension time -->', ha='center', va='center', rotation='vertical')
        plt.rcParams['axes.facecolor'] = 'black'
        plt.grid()

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot = image_from_plot[:, :, ::-1]
        plt.close()

        return image_from_plot                                        
       
        #==========================

    def eval2(self,  g_model, cur_epoch = 0 , itms = 9):                            
                                                                        
        eval_imgs =int(itms**0.5)
        row_itms = int(itms**0.5)
        z_samples = np.random.randn(eval_imgs**2, self.z_dim)
        img_array = []

        for i in range(eval_imgs):
            for j in range(eval_imgs):
                z_sample = z_samples[[i * eval_imgs + j]]
                x_sample = g_model.predict(z_sample)
                img_array.append(x_sample[0])                                  
        
        conc_grid = np.concatenate( [ np.concatenate(img_array[x:x + eval_imgs],                          
                                    axis = 1) for x in range(0,itms, eval_imgs ) ], axis = 0)        
        sim_imgs =  Image.fromarray(np.uint8(conc_grid*255))

        if self.watermark_imgs == True:                                                      
            draw = ImageDraw.Draw(sim_imgs)
            font = ImageFont.truetype("arial.ttf", 30)
            c_row, c_col, count = 0,0,0
            for in_each_row in range(row_itms):
                c_col =0
                for in_each_col in range(row_itms):                    
                    draw.text((c_row + 10 , c_col + 10),str("%.6f"%z_samples[count].mean()),(255,255,255),font=font)
                    c_col +=self.img_dim
                    count +=1
                c_row +=self.img_dim   
        
        preview_fname =   "./temp/tmp_preview.png"                      ## Nnoise_tagged_simulation_epoch-%s_[%sx%s].png"%(str(cur_epoch),str(itms),str(itms))
        sim_imgs.save(preview_fname)                                    ## save the simulation images
        try:                                                            ## sometime image preview process cause error and crash so exception handler is introduced here
        	self.loadImage(preview_fname )                              ## show the simulated image in preview screen
        	self.logger("\n[%s] | Preview of image saved @ epoch: %d"%(time_stamp(), cur_epoch) )                              
        except:
        	self.logger("\n[%s] | Warning! Preview of image saved @ epoch: %d SKIPPED due to unknown error"%(time_stamp(), cur_epoch) , color =5)

    def start_train(self):
        if self.qm.question(self,'CRISP',"Confrim contour simulation training?\nCaution: Please select proper Model ID from model id listview if needed" , self.qm.Yes | self.qm.No) == self.qm.No:
            return
        self.init_logger() 
        self.run_mode = "train_gan"
        hide_tf_warnings(supress_msg = self.warn_status)                                                      # Hide Tensorflow warning messages to make consol output clear
        self.update_vars()                                                                                    # get the latest updated variables for traning
        self.show_info()
        self.set_gan_plots()
        self.logger ("\n[%s] | Press [u]: Update preview | [s]: Save model | [c] Clear console screen [q]: Save & Quit training on CRISP preview window \n"%time_stamp() , color = 4)
        self.train_gan()

 
    def train_gan(self):
        #iters_per_sample   = 100
        self.FID_eval_freq = self.FID_eval_freq                                                               # defauult was 1000 (too frequnet so i upscaled 5x)
        total_iter         = self.traning_iter
        n_size             = self.sample_grid_dim                                                             # Def : 3 no of images in rows or column (n_size x n_szie )
        self.prev_epoch    = 0
        self.start_time    = time.time()
        trained_weights    = self.stored_model_pth +'/gan_model_' + self.model_id + "_" + str(self.mode) + '.weights'
        config_name        = self.stored_model_pth +'/gan_model_' + self.model_id + "_" + str(self.mode) + ".config"
        counter            = 0                                                          # local counter

        self.import_imgs()                                                              # import source raw images
       
        d_train_model , g_train_model , g_model = self.integrated_model()               # get the integrated models , requries g_model for generator

        #=======================
        if self.gan_train_type.count("Type_FXTH")== 1:
            self.logger("[%s] | Fixed Gradient Thresholding Training (FXTH) mode with thresholding value %d is selected "%(time_stamp(), self.grad_threshold), color = 4)
            self.logger("[%s] | All Image data will be converted to Gradient-type data"%time_stamp(), color = 4)
            self.Images = get_threshold_images(self.Images, self.grad_threshold)
        #=======================

        #=======================
        if self.gan_train_type.count("Type_AGTH")== 1:
            self.logger("[%s] | Adaptive Gradient Thresholding (AGTH) Training mode with thresholding value %d is selected "%(time_stamp(), self.adaptive_threshold), color = 4)
            self.logger("[%s] | All Image data will be converted to Gradient-type data"%time_stamp(), color = 4)
            self.Images =  adaptive_gradient_threshold(self.Images, threshold_value=self.adaptive_threshold, output_mode=self.adaptive_color, 
                                                                  ksize_x=self.adaptive_kernel, ksize_y=self.adaptive_kernel, 
                                                                  sort_option= self.adaptive_sort_mode, 
                                                                  rndseed = self.adaptive_seed, 
                                                                  show_store_plot_fname =  False)
        #=======================
        if self.gan_train_type.count("Type_NRML")== 1:
            self.logger("[%s] | Assumes as normal RGB image for GAN Training mode. No Filter is used "%time_stamp(), color = 4)
            pass


        #=======================
        if self.gan_train_type.count("Type_SRC")== 1:
            self.logger("[%s] | WARNING! GAN Train type filter is ignored. Image is taken as it is in source RGB mode "%time_stamp(), color = 5)
            pass

        img_data = img_generator(self.Images, 'gan', self.batch_size, img_dim = self.img_dim , scale_fac = self.scale_fac ).__iter__() # self.imgs
        Z        = np.random.randn(n_size**2, self.z_dim)


        def save_gan_model():                               # model autosave or store function on  keyevent
            sample_dir_name = self.preview_imgs  + "/"  +  self.model_id + "_" +  str(self.mode) \
                                                 +  "_" + self.gan_train_type[ : self.gan_train_type.find("::") ]
            check_dir(sample_dir_name)
            preview_fname =  self.model_id  +  "_"  +  str(self.mode) +  self.gan_train_type[ : self.gan_train_type.find("::") ] +  str(self.net_gan_iter) + "." + self.gan_output_type             
            self.sample(os.path.join(sample_dir_name, preview_fname), g_model, n_size, Z )
            g_train_model.save_weights(trained_weights)
            g_train_model.save_weights(trained_weights +".bkup")                                 # backup the models
            self.write_gan_summary()                                                             # write gan model summary
            self.logger("\n[%s] | Saving model @ iteration %d"%(time_stamp(),self.cur_epoch) )
            self.model_config(mod_type = "gan", act = "save", config_fname = config_name )       # save variables & configurations
            self.make_plot()                                                                     # update the plots. making plot for FID


        self.logger( "[%s] | Total items in training set: %d"%(time_stamp(),len(self.Images) ) )
        self.logger( "[%s] | Executing CRISP Res. mode  : %s pixels"%(time_stamp(),self.mode ) )               # shows teh CRISP simulation resoultionmode

        if os.path.isfile(trained_weights):
            self.logger("[%s] | Model found for model_id   : %s | mode: [%d]. Loading...."%(time_stamp(),self.model_id, self.mode) )
            g_train_model.load_weights(trained_weights)
            self.logger("[%s] | Loading configuration from : %s"%(time_stamp(),config_name))
            self.model_config( mod_type = "gan", act = "restore" , config_fname = config_name )          # load previous configuration
            self.make_plot()                                                                             # show hsitory plots
        else:
            self.logger("[%s] | Warning! No previously saved model found for model_id: %s | mode:[%d]"%(time_stamp(), self.model_id,self.mode), color =5)
            self.logger("[%s] | Starting with fresh model..."%time_stamp())

        self.write_gan_summary(write_to_file = False)                                                   # only display model summary


        if self.net_gan_iter >= total_iter:
        	self.logger("[%s] | ERROR! The models has already reached target number of iteration. Increase Iteration target to continue training..."%time_stamp(), color =5)
        	return 
            
        if self.Eval_FID_flag:
        	logs = {'fid': [], 'best': 1000}
        	self.logger ('[%s] | Initialize the FID evaluator...'%(time_stamp()))
        	fid_evaluator = FID(img_generator(self.Images, 'fid', self.batch_size, img_dim = self.img_dim, scale_fac = self.scale_fac), True) # self.imgs

        self.logger("[%s] | Initalizing generator training..."%time_stamp())


        self.logger ("\n[%s] | Press [u]: Update preview | [s]: Save model | [c] Clear terminal screen | [q]: Save & Quit training on CRISP preview window \n"%time_stamp() , color = 4)

        for self.cur_epoch in range(self.net_gan_iter, total_iter):
            for j in range(2):
                x_sample = next(img_data)
                z_sample = np.random.randn(len(x_sample), self.z_dim)
                d_loss = d_train_model.train_on_batch(  [x_sample, z_sample], None  )
            for j in range(1):
                x_sample = next(img_data)
                z_sample = np.random.randn(len(x_sample), self.z_dim)
                g_loss = g_train_model.train_on_batch(  [x_sample, z_sample], None  )
                                                     
            #============================================ very important for maintaning GUI
            key = cv2.waitKey(10) & 0xFF

            if key == ord("s"):
                self.logger("\n[%s] | Saving model requested... "%time_stamp())
                save_gan_model()
                self.eval2(g_model,cur_epoch = self.cur_epoch, itms = self.img_preview )
                self.make_plot()

            if key == ord("u"):
                self.logger("\n[%s] | Updating preview... "%time_stamp())
                img = self.evaluate(g_model, eval_imgs = self.img_preview )
                self.eval2(g_model,cur_epoch = self.cur_epoch, itms = self.img_preview )
                cv2.imshow("CRISP: preview",img)
                self.make_plot()

            if key == ord("q"):
                self.logger("\n[%s] | Exit requested. Storing current iteration & Exiting... "%time_stamp())
                save_gan_model()
                self.logger("\n[%s] | Models stored. Exiting Training...."%time_stamp())
                reset_gpu()                                                                              # reset GPU and Free memory
                return
                sys.exit(-1)


            if key == ord("c"):
                self.logger("\n[%s] | Request to clear terminal/console screen..."%time_stamp())
                try:
                	os.system('cls') if os.environ.get('OS','') == 'Windows_NT' else os.system('clear')  # for windows or linux                	                                 
                	self.logger("\n[%s] | Displaying information on cleared screen"%time_stamp())
                except:
                	pass
            #============================================

            print (Style.BRIGHT + Fore.BLUE + "\r[%s] | [ Total iter. %d | Curr. iter.: %s | D_loss: %s | G_loss: %s ] "% (time_stamp(),  # update frequently on console screen 
                																		        self.net_gan_iter, 
                	                                                                            counter, 
                	                                                                            d_loss, g_loss ), end = "" )

            if self.cur_epoch % 10 == 0:                                        # store values once in every 10 epoch
                self.DRloss_val= d_loss
                self.G_loss_val= g_loss
                self.DRloss_list.append(math.log(abs(d_loss),10))                
                self.G_loss_list.append(g_loss)

            if self.cur_epoch % self.autosave_freq == 0:
                save_gan_model()                                               # update plots
                pass
            #=================================================================================================================
            if (self.cur_epoch % self.FID_eval_freq == 0) and self.Eval_FID_flag == True:
                def _generator():                                                                                                                         # yields fake images based on batch size
                    while True:
                        _z_fake = np.random.randn(100, self.z_dim)
                        _x_fake = g_model.predict(_z_fake, batch_size=self.batch_size)
                        _x_fake = np.round((_x_fake + 1) / 2 * 255, 0)
                        _x_fake = np.array([cv2.resize(_x, dsize=(299, 299), interpolation=cv2.INTER_CUBIC) for _x in _x_fake])
                        yield _x_fake
                
                self.logger("\n[%s] | Evaluating the FID @ iteration : %d  with steps: %d"%(time_stamp(),self.cur_epoch , self.fid_steps))
                fid = round( fid_evaluator.evaluate(_generator(), True, steps= self.fid_steps), 4 )             # steps=100 for original, here  default =5           # FID spits out huge numbers for upt 4 decimal place is ok       
                self.FID_list.append((self.net_gan_iter,fid))
                
                self.logger('\n[%s] | Total Iter.: %s, Fid: %s, Mimimum (epoch, value): %s' % (time_stamp(), self.cur_epoch, fid, min(self.FID_list, 
                	                                                                                    key = lambda t: t[1])  ))                         # show min FID based on FID only
                

            if (self.cur_epoch > 10000 and self.FID_eval_freq >100) and (self.Eval_FID_flag == True):            # dynamically adjust FID frequency
                self.FID_eval_freq = 100
            #==================================================================================================================

            if (counter) % self.prev_img_freq == 0:
                img = self.evaluate(g_model, eval_imgs = self.img_preview )                                      # get matplot image as BGR image
                self.eval2(g_model,cur_epoch = self.cur_epoch, itms = self.img_preview )
                cv2.imshow("CRISP: preview",img)
                self.make_plot()                                                                                 # update plots

            if (self.cur_epoch) % 50 == 0 and self.blur_smp_flag == True :                                       # make blur sampling in every 50 epoch
            	try:
            		self.blur_sampling(g_model)
            	except:
            		self.logger("\n[%s] | Blur sampling skipped due to memory error...."%time_stamp(), color =5)
            		pass


            self.update_pbar(total_iter, self.net_gan_iter +1)
            self.net_gan_iter +=1                                                                                # count the global iteration till
            counter +=1

        self.logger("\n[%s] | Training process completed successfully. Storing Model & Exiting"%time_stamp())
        save_gan_model()
        self.make_plot()
        reset_gpu()
        return



    def gen_contours(self):                                                                                     
        if self.qm.question(self,'CRISPII',"Confrim Contour Synthesis? \nCaution: Please select proper Model ID from model id listview if needed" ,                                              # synthsize counter
                            self.qm.Yes | self.qm.No) == self.qm.No:
            return
        self.init_logger() 
        hide_tf_warnings(supress_msg = self.warn_status)                  # Hide Tensorflow warning messages to make consol output clear
        UseProcessor(self.cls_process_type)
        self.run_mode ="synthesis"                                        # must be before self.update_vars
        self.update_vars()                                                # get the latest updated variables for traning
        self.show_info()
        self.set_gan_plots()                                              # set gan plots
        self.contours_synthsizer()                                        # run contour synthsizer
        
    def contours_synthsizer(self):  
        self.n_size = self.gan_img_grid_dim                                # n x n grid size of the output image to be made
        self.z_dim  = self.z_synth_gan_dim                                 # 256 # 128

        _, g_train_model, g_model = self.integrated_model()                # Constructing models & loading trained weights. we do not need discrimnator here
        self.g_model_synthesis    = g_model
        Z = np.random.randn(self.n_size**2, self.z_dim)                    # Z-dim random vectors is randomized here

        #===================================Loading model configurations
        trained_weights = self.stored_model_pth +'/gan_model_' + self.model_id + "_" + str(self.mode) + '.weights'
        config_name     = self.stored_model_pth +'/gan_model_' + self.model_id + "_" + str(self.mode) + ".config"

        if os.path.isfile(trained_weights):
            self.logger("\n[%s] | Loading trained weights for mode: [%d]. Loading...."%( time_stamp(), self.output_shape_value))
            g_train_model.load_weights(trained_weights)
            self.logger("[%s] | Model loaded successfully....."%time_stamp())

            if os.path.isfile(config_name):
            	self.logger("\n[%s] | Model configuration found.Loading model configuration...."%time_stamp())
            	self.model_config( mod_type = "gan", act = "restore" , config_fname = config_name )                # load previous configuration
            	self.make_plot()
            	                                                                                                   # show model hsitory plots
        if not os.path.isfile(trained_weights):
        	self.logger("[%s] | ERROR! Trained generator model not found. Exiting Contour Synthesis.... "%time_stamp(), color =5)
        	return 
        #==================================

        self.write_gan_summary(write_to_file = False)                                                              # only display model summary

        self.start_time = time.time()
        prev_count =0 

        if self.gan_agument_flag:
        	self.logger("[%s] | Selected image agumentation will be applied for each simulated image(s)..."%time_stamp())

        tqdm_info= (Style.BRIGHT + Fore.BLUE + "[%s] | Generating synthetic image: "%time_stamp())
        self.gan_timestamp = str(time.time())

        sim_out_dir = self.gen_imgs_path +  "/" + "_".join([self.model_id, str(self.synth_model_dim), self.gan_timestamp ])     # keep tiem stam pas well

        check_dir(sim_out_dir)                                                     # make the dir for core simulated images

        self.logger("[%s] | Synthetic contour samples output : %s\n"%(time_stamp(), sim_out_dir), color =3)

        for cur_count in tqdm(range(self.synth_img_number), desc = tqdm_info):    

            Z = np.random.randn(self.n_size**2, self.z_dim) 	                    # IMPORT  Z+i increases intensity , Z/(i+1) decreases intensity
            C = randMat(self.n_size**2, self.z_dim, 0.75 + 0.20 * np.random.random())     

            if (self.gan_int_synth_fac != 1) or (self.gan_int_synth_fac != 0):
            	rnd_intensity = self.gan_int_synth_fac * np.random.random() # org /10
            else:
            	rnd_intensity = 0


            self.sample(sim_out_dir +"/simulated_%d_%s.%s"%(self.synth_model_dim, cur_count,self.gan_output_type),              
                                                              g_model,
                                                              self.n_size, 
                                                              (Z + rnd_intensity), 
                                                              gan_fname = str(cur_count) )
            if self.syth_intensity:
                self.sample(sim_out_dir +"/simulated_rnd_intensity_%d_%s.%s"%(self.synth_model_dim, cur_count,self.gan_output_type),          
                                                              g_model,
                                                              self.n_size, 
                                                              ( C + Z * rnd_intensity ), 
                                                              gan_fname = str(cur_count) )
        

            self.update_pbar( self.synth_img_number, cur_count +1 )                                                           

        self.make_swarm_plot()
        self.logger("[%s] | Total of %d contour(s) were successfully generated"%(time_stamp(),cur_count+1))                    
        self.logger("[%s] | Total of %d agumented image(s) were added"%(time_stamp(),self.count_agumented))


    def model_config(self, mod_type = "gan", act = "save" , config_fname = "model_config.config" , show_msg = True):

    	if mod_type == "gan":
    		obj_list = [self.net_gan_iter, self.FID_list, self.G_loss_list, self.DRloss_list ,self.blur_scr_lst, self.avg_blr_history]      			            #==========variable list to save or restoe for gan

    	if mod_type == "class":
    		obj_list = [self.net_cls_epoch, self.cls_accuracy, self.net_cls_loss , self.cls_tf_model_type, 
    		            self.cls_val_accuracy, self.net_cls_val_loss, self.cls_auroc, self.val_auroc        ]     		                                             #=========variable list to save or restoe for gan
    		#self.logger("SAVER:",self.net_cls_epoch, self.cls_accuracy, self.net_cls_loss , self.cls_tf_model_type )                                                 # just for checking if saved value exists

    	if act == "save":                                                                                                        
    		with open(config_fname, 'wb') as config_file: 
    			pickle.dump(obj_list, config_file)

    		self.logger ("[%s] | Model configuration saved...."%time_stamp() ) if show_msg == True else None


    	elif act == "restore" and os.path.isfile(config_fname):                                                                   
    		with open(config_fname, 'rb') as config_file:
    			# unpickeling var list
    			if mod_type == "gan":
    				self.net_gan_iter, self.FID_list, self.G_loss_list, self.DRloss_list, self.blur_scr_lst, self.avg_blr_history = pickle.load(config_file)

    			if mod_type == "class":
    				try:                                                                                                            
    					[self.net_cls_epoch   , self.cls_accuracy     , self.net_cls_loss , self.cls_tf_model_type,  
    				 	self.cls_val_accuracy , self.net_cls_val_loss , self.cls_auroc    , self.val_auroc ] = pickle.load(config_file)
    				except:
    					[self.net_cls_epoch   , self.cls_accuracy , self.net_cls_loss , self.cls_tf_model_type,                 
    				 	self.cls_val_accuracy, self.net_cls_val_loss  ] = pickle.load(config_file)                                 
 
    			self.logger ("[%s] | Model Configuration restored..."%time_stamp() )

    	elif act =="restore" and  (not os.path.isfile(config_fname)):
    		self.logger ("[%s] | Warning! No previous model configuration file found. Using currently available setting..."%time_stamp() , color =5)
    		pass  		


    def store_gan_config(self):
    	self.config_store()

    #============================[ CLASSIFIER BLOCK ]===================

    def class_model(self, mode = "VGG16", image_dim = 512 , cls_full_train = False , offline_flag = True):     

        if mode == "VGG16":
            from keras.applications.vgg16 import VGG16
            from keras.applications.vgg16 import preprocess_input
            fname = "./classifier_data/base_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5" if offline_flag == True else "imagenet"        
            baseModel = VGG16(weights=fname, include_top=False, input_tensor=Input(shape=(image_dim, image_dim, 3)))

        if mode == "VGG19":
            from keras.applications.vgg19 import VGG19
            from keras.applications.vgg19 import preprocess_input
            fname = "./classifier_data/base_weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5" if offline_flag == True else "imagenet" 
            baseModel =  VGG19(weights=fname, include_top=False, input_tensor=Input(shape=(image_dim, image_dim, 3)))  # use weights ='imagenet' for latest weights

        if mode == "DenseNet":
            from keras.applications.densenet import DenseNet121
            from keras.applications.densenet import preprocess_input
            fname ="./classifier_data/base_weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5" if offline_flag == True else "imagenet"       
            baseModel =  DenseNet121(weights=fname, include_top=False, input_tensor=Input(shape=(image_dim, image_dim, 3)))  # use weights ='imagenet' for latest weights

        if mode == "InceptionV3":
            from keras.applications  import InceptionV3
            from keras.applications.inception_v3 import preprocess_input
            fname ="./classifier_data/base_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5" if offline_flag == True else "imagenet" 
            baseModel =  InceptionV3(weights=fname, include_top=False, input_tensor=Input(shape=(image_dim, image_dim, 3)))  

        if mode == "ResNet50":
            from keras.applications import ResNet50
            from keras.applications.resnet50 import preprocess_input
            fname ="./classifier_data/base_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5" if offline_flag == True else "imagenet" 
            baseModel =  ResNet50(weights=fname, include_top=False, input_tensor=Input(shape=(image_dim, image_dim, 3)))  

        #the base model
        if self.show_cls_summary:
            self.logger("[INFO] Summary for base %s model..."%mode)
            self.logger(baseModel.summary())

        #Classifier traning data categories (each folderdler is annoted one chategory)
        train_categories = glob.glob(self.cls_train_data + "/Train/*")                        # useful for getting number of classes (each sub dir is a class)
        self.cls_labels  = [ os.path.basename(folder) for folder in train_categories ]        # keep reocrd of the trainig clases IDs
        class_labels_str =  ",".join([x for x in self.cls_labels])
        self.logger("[%s] | Class Lables: %s"%(time_stamp(), class_labels_str), color = 3)

        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)                 
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(image_dim, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(len(train_categories), activation="softmax")(headModel)  # output number of chategories by model    
        model = Model(inputs=baseModel.input, outputs=headModel)                   # place the head FC model on top of the base model (this will become the actual model we will train)
        
        # loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
        for layer in baseModel.layers:                                             # do not train the entire model from scratch (you may if you like, then set layers.trainable = True)
            layer.trainable = cls_full_train                                       # TODO: False by default, making this True will train whole model from start

        # loop over all layers in the base model and freeze them so they will *not* be updated during the first training process

        if len(train_categories) == 2 and self.run_mode == "classifier_training":
            model.compile( loss='categorical_crossentropy', optimizer=  Adam(lr=self.cls_lr, beta_1= 0.5) ,  metrics=['accuracy', auroc] )  if self. cls_optimizer == "Adam" else None
            model.compile( loss='categorical_crossentropy', optimizer=  RMSprop(lr=self.cls_lr)  ,  metrics=['accuracy', auroc] )  if self.cls_optimizer == "RMSprop" else None
            model.compile( loss='categorical_crossentropy', optimizer=  Adadelta(lr=self.cls_lr) ,  metrics=['accuracy', auroc] )  if self.cls_optimizer == "Adadelta" else None     
            model.compile( loss='categorical_crossentropy', optimizer=  Adamax(lr=self.cls_lr)   ,  metrics=['accuracy', auroc] )  if self.cls_optimizer == "Adamax" else None     
            model.compile( loss='categorical_crossentropy', optimizer=  Nadam(lr=self.cls_lr)    ,  metrics=['accuracy', auroc] )  if self.cls_optimizer == "Nadam" else None   #, decay= round(1e-2/epochs)        
            model.compile( loss='categorical_crossentropy', optimizer=  Nadam(lr=self.cls_lr)    ,  metrics=['accuracy', auroc] )  if self.cls_optimizer == "SGD" else None     #, decay= round(1e-2/epochs)  
        else:
            # cannot defien ROC in case of more than 2 classes
            model.compile( loss='categorical_crossentropy', optimizer=  Adam(lr=self.cls_lr, beta_1= 0.5) ,  metrics=['accuracy'] )  if self. cls_optimizer == "Adam" else None
            model.compile( loss='categorical_crossentropy', optimizer=  RMSprop(lr=self.cls_lr)  ,  metrics=['accuracy'] )  if self.cls_optimizer == "RMSprop" else None
            model.compile( loss='categorical_crossentropy', optimizer=  Adadelta(lr=self.cls_lr) ,  metrics=['accuracy'] )  if self.cls_optimizer == "Adadelta" else None     
            model.compile( loss='categorical_crossentropy', optimizer=  Adamax(lr=self.cls_lr)   ,  metrics=['accuracy'] )  if self.cls_optimizer == "Adamax" else None     
            model.compile( loss='categorical_crossentropy', optimizer=  Nadam(lr=self.cls_lr)    ,  metrics=['accuracy'] )  if self.cls_optimizer == "Nadam" else None   #, decay= round(1e-2/epochs)        
            model.compile( loss='categorical_crossentropy', optimizer=  Nadam(lr=self.cls_lr)    ,  metrics=['accuracy'] )  if self.cls_optimizer == "SGD" else None     #, decay= round(1e-2/epochs)  

        return model


    def make_cls_plot(self):                                                      # mode real tiem plot for classifier
       	try:
       		self.class_accuracy_plot.removeItem(self.gfx3)
       		self.class_accuracy_plot.removeItem(self.gfx4)
       	except:
       		pass
        self.datos  = pg.ScatterPlotItem()
        self.gfx3   = self.class_accuracy_plot.addPlot(title='classifier accuracy')
        self.gfx3.setLabel('left','class. accuracy (%)')
        self.gfx3.setLabel('bottom','epoch(s)')
        self.datos = self.gfx3.plot(pen='y')
        self.datos.setData(self.cls_accuracy)
        self.gfx3.enableAutoRange(x=True,y=True)
        self.datos  = pg.ScatterPlotItem()
        self.gfx4   = self.class_accuracy_plot.addPlot(title='classifier loss')
        self.gfx4.setLabel('left','class. loss')
        self.gfx4.setLabel('bottom','epoch(s)')
        self.datos = self.gfx4.plot(pen='y')
        self.datos.setData(self.net_cls_loss)
        self.gfx4.enableAutoRange(x=True,y=True)

        self.gan_current_loss_display.insertPlainText("\ncurr. epoch: %d | training acc.:%s%% | training loss: %s"%(self.net_cls_epoch, 
                                                                                                                    str(self.cls_accuracy[-1:]),
                                                                                                                    str(self.net_cls_loss[-1:])  ) )  # get the last updates value loss
        #self.gan_current_loss_display.insertPlainText("\nCurr. Training loss:" + )  # get the last updates value loss
        time.sleep(0.3)


    def val_graph_show(self):

        if len(self.cls_auroc) >1:                  # show best AUROC v/s epoch

            fig, axes = plt.subplots(nrows=4, ncols=1)
            axes[0].plot(self.cls_val_accuracy)
            #axes[0].set_title("validation accuracy")
            axes[0].set_ylabel ("val. acc (%)")

            axes[1].plot(self.net_cls_val_loss)         
            #axes[1].set_title("validation loss")
            axes[1].set_ylabel ("val. loss")

            fig, axes = plt.subplots(nrows=4, ncols=1)
            axes[0].plot(self.cls_val_accuracy)
        	#axes[0].set_title("validation accuracy")
            axes[0].set_ylabel ("val. acc (%)")

            axes[1].plot(self.net_cls_val_loss)        	
        	#axes[1].set_title("validation loss")
            axes[1].set_ylabel ("val. loss")

            axes[2].plot(self.cls_auroc)
        	#axes[2].set_title("classification AUROC")
            axes[2].set_ylabel ("class. auroc")     	

            axes[3].plot(self.val_auroc)
        	#axes[3].set_title("validation AUROC")
            axes[3].set_ylabel ("val. auroc")

            axes[3].set_xlabel ("epoch(s)")                                   # common plotting eoch labe;

        else:                                                                 # do not show AUROC only show training and validation accuracy
        	fig, axes = plt.subplots(nrows=2, ncols=1)
        	axes[0].plot(self.cls_val_accuracy)
        	#axes[0].set_title("validation accuracy")
        	axes[0].set_ylabel ("class. accuracy (%)")

        	axes[1].plot(self.net_cls_val_loss)        	
        	#axes[1].set_title("validation loss")
        	axes[1].set_ylabel ("val. loss")
        	axes[1].set_xlabel ("epoch(s)")


        plt.rcParams['axes.facecolor'] = 'black'
        plt.tight_layout()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot = image_from_plot[:, :, ::-1]
        plt.close('all')
        self.make_cls_plot()
        cv2.imshow("CRISP: preview",image_from_plot)


    def train_class(self):
        if self.qm.question(self,'CRISPII',"Confrim Classifier training? \nCaution: Please select proper Model ID from model id listview if needed" , self.qm.Yes | self.qm.No) == self.qm.No:
            return
        self.init_logger() 
        self.run_mode = "classifier_training"
        self.update_vars()
        hide_tf_warnings(supress_msg = self.ignore_cls_warnings)
        UseProcessor(self.cls_process_type)
        self.show_info()
        self.train_classifier()      


    def train_classifier(self):
        # re-size all the images to this
        image_dim  = self.cls_input_dim  #512                                     # input dimesion of classifier source image
        mode       = self.cls_tf_model_type                                       # "vgg16"
        self.classifier_model = self.cls_model_path +"/contour_features_model_" + self.class_model_id + "_"  + str(self.cls_input_dim) + "pxl_" +  mode + ".h5"                 # the full pathmane of classifer model
        self.classifier_config= self.cls_model_path +"/contour_features_model_" + self.class_model_id + "_"  + str(self.cls_input_dim) + "pxl_" +  mode + ".config"             # configuration name of classifier
        self.start_time = time.time()
        self.cur_val_multi_auroc = 0.000
        self.cur_epoch  = 0
        cls_lr_history  = []

        self.multi_roc_file_path  =  self.cls_model_path + "/scores_" + self.class_model_id + "_"+ str(image_dim)+"pxls_"+ mode + ".csv"

        def step_decay(epoch, lr ):                                                # custom learning rate adaptive reduction module
        	self.net_cls_epoch +=1                                                 # todo : ranme to self.net_cls_epoch
        	self.cur_epoch      = epoch
        	#============================= For AUROCs (binary or multiclass AUROCs)
        	        	
        	if (self.cur_epoch )  % 5 == 0:  # for validation cohort do for all epoch
        		# Generate predictions
        		pred_probs = model.predict_generator(self.valid_set)
        		true_labels = self.infer_set.classes
        		cls_labels = list(self.infer_set.class_indices.keys()) 


        		# Calculate AUROC
        		if self.valid_set.class_mode == 'binary':
        			auroc = roc_auc_score(true_labels, pred_probs)
        			self.kappa    = cohen_kappa_score(true_labels, pred_probs)
        			self.f1score  = f1_score(true_labels, predictions_lables, average='weighted')
        		else:
        			inf_fnames =  [os.path.join(self.infer_set.directory, fname) for fname in self.infer_set.filenames]              # important Bug FIXED for Test or COhort fulenames

        			img_data = img_convert_for_inference(inf_fnames,self.cls_input_dim)            
        			predictions, predictions_lables =[], [] 
        			for index, each_target in enumerate(img_data):
        			     result = model.predict(each_target)   
        			     predictions.append (result) # return result
        			     predictions_lables.append(int(np.argmax(result, axis=1)))
        			#===============
        			val_loss, val_accuracy  =  model.evaluate_generator(self.infer_set, steps= self.infer_set.samples // self.infer_set.batch_size)
        			print(f"\nModel loss     : {val_loss: <20}\nModel accuracy : {val_accuracy:<20}") # loss ,accuracy, auroc # output
        			auroc_scores, gan.cur_val_multi_auroc = plot_multi_class_auroc(true_labels, predictions_lables , cls_labels , show_plt = False)

        			# Calculate Kappa After making predictions
        			self.kappa    = cohen_kappa_score(true_labels, predictions_lables)
        			self.f1score  = f1_score(true_labels, predictions_lables, average='weighted')
        
      
        		#=========Compute model robestness
        		self.compute_robustness(model)
        		#==============================

        	#================================================================show updated updates values
        	print(Style.BRIGHT + Fore.BLUE + "\r\r[%s] | [net epoch:%d|Cur. epoch:%d|class. auroc:%0.5f|val_acc.:%0.5f|val_loss:%0.5f|val_auroc:%0.5f]"% ( time_stamp(), 
        		                                                                                       self.net_cls_epoch,
        		                                                                                       epoch+1, 
                                                                                                       self.cur_cls_auroc,
        		                                                                                       self.cur_cls_val_acc,
        		                                                                                       self.cur_cls_val_loss,
                                                                                                       self.cur_val_auroc),
        		                                                                                       end =""  )

            #============================================ very important for maintaning GUI       	
        	if (self.cur_epoch % 10 == 0):                                                                           # show the graphs whil model is saved
        		self.make_cls_plot() 

        	key = cv2.waitKey(10) & 0xFF

        	if  key == ord("s"):
        		self.logger("\n[%s] | Saving model requested... "%time_stamp())
        		self.logger("[%s] | Model weights and configuration saved @ current epoch %d"%(time_stamp(), epoch)  )
        		model.save(gan.classifier_model)
        		self.model_config( mod_type = "class", act = "save" , config_fname = gan.classifier_config ,show_msg = True)
        		self.write_class_summary()

        	if  key == ord("u"):
        		self.logger("\n[%s] | Updating the graphs... "%time_stamp())
        		self.make_cls_plot()
        		self.val_graph_show() if self.popup_win_flag  else None

        	if  key == ord("q"):
        		self.logger("\n[%s] | Exit requested. Saving the model and Exiting... "%time_stamp())
        		model.save(gan.classifier_model)
        		self.model_config( mod_type = "class", act = "save" , config_fname = gan.classifier_config ,show_msg = True)
        		self.write_class_summary()
        		self.logger("[%s] | Model and Configuration saved. Exiting.."%time_stamp())
        		reset_gpu()
        		del gan.classifier_model

        	if key == ord("c"):
        		self.logger("\n[%s] | Request to clear terminal/console screen..."%time_stamp())
        		try:
        			os.system('cls')  if os.environ.get('OS','') == 'Windows_NT' else os.system('clear')  # for windows or linux                                      # for windows
        			self.logger("\n[%s] | Displaying information on cleared screen"%time_stamp())
        		except:
        			pass

        	#====================================================================
        	if self.popup_win_flag  and (self.cur_epoch  % 10 == 0) :                                      # updates may not owrk in absensce of this option
        		self.val_graph_show()    

        	#================================================================Store
        	if self.net_cls_epoch % 100 == 0:
        		eval_path ="./classifier_data/evaluations/" + self.class_model_id + "_" + str(mode)
        		check_dir(eval_path)                                                                       # make dir if not exists
        		plt.savefig(eval_path + "/" + self.class_model_id + "_" + mode + "_epoch_" +str(self.net_cls_epoch) +".png")
        	#=================================================================

        	if (epoch +1) % self.cls_autosave_freq == 0:
        		self.logger("\n[%s] | Saving model weights..."%time_stamp())
        		model.save(self.classifier_model)
        		self.logger("[%s] | Model weights and configuration saved @ current epoch %d"%(time_stamp(), epoch)  )
        		self.model_config(mod_type = "class", act = "save" , config_fname = gan.classifier_config ,show_msg = True)
        		self.write_class_summary()

        	if not self.cls_lr_decay_flag:
        		return lr

        	self.update_pbar(self.cls_train_epoch, self.net_cls_epoch)

        	if (epoch+1) % self.cls_lr_decay_freq == 0:
        		drop  = round(self.cls_lr_decay_rate/100,2)                                     # convert percentage to float i.e 25% == 0.25
        		lrate = self.cls_lr * math.pow( drop, ((epoch+1)//self.cls_lr_decay_freq)  )    # makes exponential decline in lr
        		self.logger ("[%s] | [ lr %f decayed by %f%% @ epoch %d to %f ]"% (time_stamp(), lr,self.cls_lr_decay_rate,epoch, lrate) )
        		cls_lr_history.append((lr, epoch))  
        		return lrate
        	else:
        		return lr   
        	#===============================================================                                                                           

        lrate = LearningRateScheduler(step_decay)
      
        class Class_AccuracyHistory(keras.callbacks.Callback):
        	def __init__(self):
        		pass

        	def on_train_begin(self, logs={}):
        		pass

        	def on_epoch_end(self, batch, logs={}):
        		self.cur_cls_val_acc  = gval(logs.get('val_acc'))
        		self.cur_cls_val_loss = gval(logs.get('val_loss'))
        		self.cur_cls_acc      = gval(logs.get('acc'))
        		self.cur_cls_loss     = gval(logs.get('loss'))
        		self.cur_val_auroc    = gval(logs.get('val_auroc'))
        		self.cur_cls_auroc    = gval(logs.get('auroc'))
        		gan.cur_cls_val_acc   = self.cur_cls_val_acc
        		gan.cur_cls_val_loss  = self.cur_cls_val_loss
        		gan.cls_accuracy.append(round(self.cur_cls_acc * 100, 5) )
        		gan.net_cls_loss.append(self.cur_cls_loss)
        		gan.cls_val_accuracy.append(round(self.cur_cls_val_acc * 100,5) )
        		gan.net_cls_val_loss.append(self.cur_cls_val_loss)

        		#=======================================
        		if len(gan.cls_labels) == 2:
        			gan.cur_val_auroc = self.cur_val_auroc
        			gan.cur_cls_auroc = self.cur_cls_auroc
        		else:
        			gan.cur_val_auroc = gan.cur_val_multi_auroc   # now uses the OneAgaintRest average inference auroc
        			gan.cur_cls_auroc = gan.cur_val_multi_auroc   # for now both val & classification multi_auroc are same for multi aurocs               

                	# append the new AUROCs
        		gan.cls_auroc.append(gan.cur_cls_auroc)
        		gan.val_auroc.append(gan.cur_val_auroc)
        		#======================================


        self.logger("\n==========================================================")
        self.logger("\n[%s] | Initating model training for: %s"%(time_stamp(), mode))
        self.logger("\n[%s] | Warning! if more than 2 classes exists, the AUROCs will be displayed as Average AUROCs and \nIndivisual AUROCs will be logged (and displayed)"%time_stamp(), color =5)

        # set the model and try to train
        model = self.class_model(mode = self.cls_tf_model_type, image_dim = self.cls_input_dim , cls_full_train =False , offline_flag = self.use_offline_model)

        if not os.path.isfile(self.classifier_model):
        	self.logger("[%s] | No previous model found for %s ! Starting from base weights"%(time_stamp(), mode) )
       
        if os.path.isfile(self.classifier_model):
        	self.logger("[%s] | Previously saved weights found for %s model. Loading weights..."%(time_stamp(), mode))
        	model.load_weights(self.classifier_model)
        	self.model_config( mod_type = "class", act = "restore" , config_fname = self.classifier_config)       # load previous configuration i.e: (self.net_cls_epoch, self.cls_accuracy, self.net_cls_loss , self.cls_tf_model_type )
    
        #=======================================================================================================    
        self.write_class_summary(write_to_file =False)                                                            # show classifier information
        #=======================================================================================================

        train_datagen = ImageDataGenerator( rotation_range= self.cls_img_rot_slider_value,    # 5 default
        	                                rescale = 1./255,                                 # fixed
        	                                width_shift_range=0.01,                           # fixed
        	                                height_shift_range=0.01,                          # fixed
                                            shear_range = self.cls_img_shear_slider_value,    # 0.2 default
                                            zoom_range = self.cls_img_zoom_slider_value,      # 0.15 default
                                            vertical_flip = self.cls_agu_vflip_flag_value,    # False default
                                            horizontal_flip = self.cls_agu_hflip_flag_value,  # False default
                                            fill_mode= self.cls_set_img_fill_mode)            # "nearest"  default

        test_datagen = ImageDataGenerator(rescale = 1./255)

        training_set = train_datagen.flow_from_directory(   os.path.join(self.cls_train_data, "Train"),                #  './classifier_data/Datasets/Train',
                                                            target_size = (image_dim, image_dim),
                                                            batch_size = self.cls_batch_size,
                                                            class_mode = 'categorical')

        self.valid_set =  test_datagen.flow_from_directory(   os.path.join(self.cls_train_data, "Test"),                     # './classifier_data/Datasets/Test',
                                                       target_size = (image_dim, image_dim),
                                                       batch_size = self.cls_batch_size,
                                                       class_mode = 'categorical')



        # agressive for image datagen for sample robustness factor calculation 
        self.perturbation_datagen = ImageDataGenerator(rescale=1./255,
                                                    brightness_range=[0.7, 1.3],
                                                    rotation_range= 2,
                                                    width_shift_range=0.03,
                                                    height_shift_range=0.02  )



        if not self.use_validation_cohot_scores:
            self.infer_set =  test_datagen.flow_from_directory(os.path.join(self.cls_train_data, "Inference_samples"), # './classifier_data/Datasets/Inference_samples',
                                                       target_size = (image_dim, image_dim),
                                                       batch_size = self.cls_batch_size,
                                                       class_mode = 'categorical')

            self.perturbed_val_generator = self.perturbation_datagen.flow_from_directory(  os.path.join(self.cls_train_data, "Inference_samples"),
                                                        target_size=(image_dim, image_dim),
                                                        batch_size=self.cls_batch_size,
                                                        class_mode='categorical',
                                                        shuffle=False )

        else:
            self.logger("[%s] | WARNING! User has enabled validation cohort to compute scores during training. Please make sure the validation cohort dataset is structured same as train or test set"%time_stamp(), color =5)
            self.logger("[%s] | NOTE: It may slowdown the training session if large numers of samples are present in validation set. These scores has no effect on model training. \nIts is just to live preview progress of training and model performance"%time_stamp(), color =5)
            
            self.infer_set =  test_datagen.flow_from_directory(self.cls_testing_path,                                    # './datasets/Cohort-II',
                                                       target_size = (image_dim, image_dim),
                                                       batch_size = self.cls_batch_size,
                                                       class_mode = 'categorical')

            self.perturbed_val_generator = self.perturbation_datagen.flow_from_directory( self.cls_testing_path,
                                                        target_size=(image_dim, image_dim),
                                                        batch_size=self.cls_batch_size,
                                                        class_mode='categorical',
                                                        shuffle=False )     



        
        #print("\nValidation set classes : " , self.valid_set.classes)
        class_pairs = [ ab for ab in itertools.permutations(range(self.valid_set.num_classes),2) if ab[0] < ab[1] ] # avoid (a,b) = (b,a) # list(itertools.permutations(range(self.valid_set.num_classes), 2) ) 

        # Showp class labels and codes being trained

        if True:
            self.logger("Class name[class code] -vs- Class mame[class code]", color =3)
            self.logger("===========================================================")
            for a,b in class_pairs:
                self.logger(self.cls_labels[a]+ "[%d]  -vs-  "%a + self.cls_labels[b]+"[%d]"%b , color =2)
            self.logger("===========================================================")

        self.class_pairs_header = "epoch," + \
                                  ",".join([ self.cls_labels[a] + "-vs-" + self.cls_labels[b] for (a,b) in class_pairs]) + \
                                  ",Avg_AUROC"  # add average at end



        history = Class_AccuracyHistory()                                                               # this is the class within thsi function to calculate the prediction

        if self.net_cls_epoch >= self.cls_train_epoch:
        	self.logger("[%s] | Warning! The Classifier models has already reached target number of iteration...\n Increase Iteration target to continue training. Exiting..."%time_stamp(), color = 5)
        	return 

        # fit /train the model
        r = model.fit_generator(training_set,
                                        validation_data=self.valid_set,
                                        epochs= self.cls_train_epoch,  
                                        steps_per_epoch=len(training_set),
                                        validation_steps=len(self.valid_set), 
                                        callbacks=[history, lrate ],
                                        verbose= self.class_verbosity) #, checkpointer])

        self.logger("[%s] | Saving trained weights"%time_stamp())
        model.save(self.classifier_model)
        self.model_config( mod_type = "class", act = "restore" , config_fname = self.classifier_config)     # load previous configuration
        self.logger("[%s] | Weights & configurations saved successfully..."%time_stamp())

        # loss
        plt.plot(r.history['loss'], label='train loss')
        plt.plot(r.history['val_loss'], label='val loss')
        plt.legend()
        plt.savefig('./classifier_data/evaluations/LossVal_loss_'+ mode + ".png")

        # accuracies
        plt.plot(r.history['acc'], label='train acc')
        plt.plot(r.history['val_acc'], label='val acc')
        plt.legend()
        plt.savefig('./classifier_data/evaluations/AccVal_acc_' + mode + ".png")
        plt.close()

    def run_class(self):
        if self.qm.question(self,'CRISPII',"Confrim run Classifier? \nCaution: Please select proper Model ID from model id listview if needed"  , self.qm.Yes | self.qm.No) == self.qm.No:
            return
        self.run_mode = "classifier_inference"
        self.init_logger() 
        self.update_vars()
        hide_tf_warnings(supress_msg = self.ignore_cls_warnings)
        UseProcessor(self.cls_process_type)
        self.show_info()
        self.classifier()      

    def update_pbar(self,tot_index, cur_index):
        ETA=(tot_index - cur_index) * (time.time()-self.start_time)/(2*60)
        itm_per_sec =  2/(0.0001 + time.time()-self.start_time)                        # rough ETA where 0.0001 is epselon for time.time()
        self.start_time = time.time()
        self.progressBar.setValue(round(100 * (cur_index + 1 ) /tot_index) )
        self.progress_status.setText("status: %s network | ETA(min): %0.5f  | avg. epoch(Iter.)/s : %0.4f "%(self.run_mode, ETA, itm_per_sec) ) 

    def classifier(self):                                                              # Classifier protion of teh CRISP
    	#==========================================Make report file if selected
    	if self.write_report_file == True:
    		check_dir(self.classifier_report_path.toPlainText().strip())   # make dir if not exists
    		file = open (self.report_filepath    + "/"  +  
    			         self.class_model_id     + "_" +  
    			         self.cls_tf_model_type  + "_" +  
    			         str(self.cls_input_dim) + "." + 
    			         self.report_filetype  ,   "w" )
    		file.write  (self.para_info )


    		if self.report_filetype   == "CSV":                         # Comma ' to  seperate 
                	seperator  = ","                
    		elif self.report_filetype == "TSV":                         # tab ord(34) to seperate 
                	seperator  = "  "                                  
    		elif self.report_filetype == "TXT":                         # no seperation.
                	seperator  = None


    		if seperator != None:
                	header_info =  seperator.join(["SN.", "Image_source", "Inferred_Class", "Confidence"]) 
    		else: 
                	header_info = "#Processed inference results"

    		file.write ("\n" + header_info)

    	#===========================================
    	mode       = self.cls_tf_model_type 
    	self.classifier_model = self.cls_model_path +"/contour_features_model_" + self.class_model_id + "_"  + str(self.cls_input_dim) + "pxl_" +  mode + ".h5"                 # the full pathmane of classifer model
    	self.classifier_config= self.cls_model_path +"/contour_features_model_" + self.class_model_id + "_"  + str(self.cls_input_dim) + "pxl_" +  mode + ".config"             # configuration name of classifier

    	self.logger("[%s] | Running classifier module for inferencing..."%time_stamp())
    	img_width, img_height = (self.cls_input_dim,self.cls_input_dim)  # (512,512) default

    	self.logger("[%s] | Loading trained model..."%time_stamp())
    	model =  load_model(self.classifier_model, compile =False)                                                                          # Compile MUST be set to false or else auroc metric eror will come
    	model.compile( loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy', auroc] )                                   # mist be compiling auroc here              

    	self.model_config( mod_type = "class", act = "restore" , config_fname = self.classifier_config)                                    # restore model hsitory
    	self.make_cls_plot()                                                                                                               # plot history

    	#=======================================================================================================    
    	self.write_class_summary(write_to_file =False)                                                                                     # show classifier information
    	#=======================================================================================================

    	try:
        	with open(self.cls_model_path +"/"+ self.class_model_id +".model" , 'r') as model_id:   # read model ID file for classification labels
        		self.cls_labels = model_id.read().strip().split(",")                                # read as string and split ot make a list
    	except:
        	self.cls_labels = None
        	self.logger('[%s] Warning! Unable to read the model abels. Please refer to the classification code in corrsponding tarining dataset'%time_stamp(), color = 5 )

                            
    	#preprocess image
    	self.logger("#----------------------------------------\n")
    	if self.search_subfolders == True:
        	target_list = find_image_files(self.cls_testing_path)
        	self.logger("[%s] | Scanning all folders and subfolder(s)"%time_stamp(), color = 3)
    	else:
        	target_list = get_source_images(self.cls_testing_path)                               # do not suffle as it is related to index in img_data
        	self.logger("[%s] | Scanning single folder"%time_stamp(), color =3)

    	self.logger("[%s] | Total Contour(s) found: %d"%(time_stamp(), len(target_list)))
        
    	img_data = img_convert_for_inference(target_list,self.cls_input_dim)
        #=================================================================      
    	if (self.resize_watermarked and self.put_watermark):                                   # if put watermark and resize image
        	self.logger("[%s] | Watermarked contours will be resized to (w x h): (%d, %d)"%(time_stamp(), self.watermark_img_width, self.watermark_img_width))        
    	
    	self.start_time =time.time()
    	prev_count =0
    	plt.ion() if not self.pause_each_heatmap else None                                      #
    	for index, each_target in enumerate(img_data):
        	result = model.predict(each_target)                                              # gethe model predicted value
        	pred = np.argmax(result, axis=1)                                                 # return result
        	confidence = result[0][pred] * 100                                               # get the confiudence in 100%

        	display_color = 2 if confidence > self.inference_thrshold else 5                                        # if confidence is greater than minimum thresthold:
        	self.logger("File No.: %d] %s --> %s [Code: %d]| confidence: %0.2f%% "%(index, target_list[index], self.cls_labels[int(pred)] , pred, confidence ), color = 2 )
        	
        	# write to file        	
        	file.write ("\n"+ out_info) if (self.write_report_file == True and header_info == None) else None      # write to file for Text format

        	if (self.write_report_file == True  and header_info != None):
        		sv_info  = seperator.join([str(index), str(target_list[index]), self.cls_labels[int(pred)] + " [Code: %s]"%str(pred), str(confidence)])       # write to file    fot TSV or CSV format
        		file.write ("\n"+ sv_info)

        	if self.put_watermark:
        		marked_fname = self.classifier_report_path.toPlainText().strip() + "/watermarked_" + self.classifier_model_id.toPlainText().strip()                           #make folder for water marking
        		check_dir(marked_fname)                                                                             # check dire and makes if necessary
        		watermark = " [GRASP inference]\n Source %d : %s\n Inference Class : %s (code: %d) \n Confidence  : %s%%"%(index, target_list[index],  self.cls_labels[int(pred)], pred, confidence)
        		watermark_inference(input  = target_list[index],
                                    output = marked_fname + "/watermarked_" + os.path.basename(target_list[index]), 
                                    text   = watermark,
                                    resize_flag= self.resize_watermarked,
                                    dim = (self.watermark_img_width, self.watermark_img_height))
            
        	if self.show_pred_heatmap:
        		check_dir(os.path.join(self.cls_heatmap_fpath, "heatmap_" + self.class_model_id))
        		heatmap_file = os.path.join(self.cls_heatmap_fpath, "heatmap_"+ self.class_model_id , "heatmap_" + self.cls_heatmap_level+"_" + os.path.basename(target_list[index]))
        		self.prediction_heatmap(img =each_target ,model = model, target_layer_index = self.cls_heatmap_index , out_fname = heatmap_file ) #'block5_conv2'

        	self.update_pbar(len(img_data), index)

    	self.logger("[%s] | Inferencing completed..."%time_stamp())   
    	self.logger("[%s] | Report written to   :%s"%(time_stamp(),self.report_filepath))  if self.write_report_file == True else None 
    	self.logger("[%s] | Watermarked results :%s"%(time_stamp(),marked_fname))  if self.put_watermark == True else None
    	plt.ioff() if not self.pause_each_heatmap else None
    	self.classfier_auc(model)

    def classfier_auc(self, model ,val_generator = None):
        if val_generator == None:
            test_datagen = ImageDataGenerator(rescale = 1./255)

        if not self.use_validation_cohot_scores:
            val_generator =  test_datagen.flow_from_directory(os.path.join(self.cls_train_data, "Inference_samples"), # './classifier_data/Datasets/Inference_samples',
                                                       target_size = (self.cls_input_dim, self.cls_input_dim),
                                                       batch_size = self.cls_batch_size,
                                                       class_mode = 'categorical')
        else:
            self.logger("[%s] | WARNING! User has choosen validation cohort to compute scores during training.\nPlease make sure the dataset is structured same as Train or Test set"%time_stamp(), color =5)

            val_generator =  test_datagen.flow_from_directory(self.cls_testing_path, # './classifier_data/Datasets/Inference_samples',
                                                       target_size = (self.cls_input_dim, self.cls_input_dim),
                                                       batch_size = self.cls_batch_size,
                                                       class_mode = 'categorical')     



        # Evaluate the model
        cls_labels  =list(val_generator.class_indices.keys()) #[ str(labels) for labels in val_generator.class_indices.keys()]
        org_fnames  =val_generator.filenames 
        self.logger("\nUnique classes : %s"%cls_labels, color = 3)
        #true_classes = val_generator.classes                                                                                # gives in terms of codes
        true_classes= [filename.split(os.path.sep)[-2] for filename in val_generator.filenames]                              # gives the foldername of the sub classes

        if not self.use_validation_cohot_scores:
            inf_fnames =  [os.path.join(self.cls_train_data,"Inference_samples",fnam) for fnam in val_generator.filenames]   # get all full path of filenames for default
        else:
            inf_fnames =  [os.path.join(self.cls_testing_path, fnam) for fnam in val_generator.filenames]                    # get all full path of filenames in  yesy custom path
            

        img_data = img_convert_for_inference(inf_fnames,self.cls_input_dim)

        self.start_time =time.time()
        prev_count =0
        predictions =[]                                                                                      # holds teh results as multi-array RAW from model
        predictions_lables =[]                                                                               # holdes the predicted class as real names strings
        for index, each_target in enumerate(img_data):
            result = model.predict(each_target)   
            predictions.append (result) # return result
            predictions_lables.append(int(np.argmax(result, axis=1)))
            #print(os.path.basename(inf_fnames[index]), cls_labels[int(np.argmax(result, axis=1))])                       

        #===============
        self.logger("Model evaluation scores", color = 3)
        val_loss, val_accuracy , val_auroc =  model.evaluate_generator(val_generator, steps= val_generator.samples // val_generator.batch_size)
        print(f"Model loss     : {val_loss: <20}\nModel accuracy : {val_accuracy:<20}\nModel Avg.AUROC: {val_auroc:<20}") # loss ,accuracy, auroc # output
        #========ss======  
        #predictions =  model.predict_generator(val_generator)                                                   # Make predictions on the test set
        #predictions_lables =  np.argmax(predictions, axis=1)

        self.logger("\nInference source: %s"%os.path.join(self.cls_train_data,"Inference_samples"), color = 3)
        self.logger("\nIndex        |        Inference sample        |        Predicted        |        True class        |        Confidence")
        print("============================================================================================================")
        for index,(pred_,true_) in enumerate(zip(predictions_lables,true_classes)):
            confidence = float(np.max(100* predictions[index]))
            display_color = 2 if confidence > self.inference_thrshold else 5 
            self.logger(f"{index:<6} {org_fnames[index]:<50} {cls_labels[pred_]:<20} {true_:<20} {np.max(100* predictions[index]):0.2f}%", color =display_color)
        print("============================================================================================================")
    
        # COMPUTING BASIC STATISTICS
        true_values        = val_generator.classes
        predicted_values   = predictions_lables

        if self.show_inference_stats:                                                                       # show simple statsitics
            true_labels      = [cls_labels[class_code] for class_code in true_values]
            predicted_labels = [cls_labels[class_code] for class_code in predicted_values]
            self.classification_analysis(true_labels,predicted_labels)                                      # Show statistics

        self.logger("\n[%s] | Computing AUROC in OneVsRestClassifier scheme for Validation cohot :"%time_stamp() )               # COMPUTE oNE VERSUS REST ROCs

        auroc_scores, average_auroc = plot_multi_class_auroc(true_values, predicted_values , cls_labels)


    def compute_robustness(self, model):
        # This factor measures how much the model's performance drops when faced with these perturbations. 
        # A higher RF (closer to 1) indicates that the model is robust and does not lose much accuracy under slight variations.

        clean_loss, clean_accuracy  =  model.evaluate_generator(self.infer_set , steps= self.infer_set .samples // self.infer_set .batch_size, verbose = 0)

        perturbed_loss, perturbed_accuracy = model.evaluate_generator(self.perturbed_val_generator, steps= self.perturbed_val_generator.samples // self.perturbed_val_generator.batch_size,verbose = 0)

        self.rf_score = round((perturbed_accuracy / clean_accuracy) ,4)
        
        self.logger("-----------------------------------", color = 2)   
        self.logger(f"F1 score                 : {round(self.f1score,4)}"    , color = 4) 
        self.logger(f"Kappa score              : {round(self.kappa,4)}"      , color = 4)            
        self.logger(f"Clean val. accuracy      : {round(clean_accuracy,4)}"  , color = 4) 
        self.logger(f"Perturbed val. accuracy  : {round(perturbed_accuracy, 4)}", color = 4)
        self.logger(f"Robustness Factor (RF)   : {round(self.rf_score,4)}"      , color =4)
        self.logger("-----------------------------------", color = 2)

        return self.rf_score


 

    def Exit_CRISP(self):
        if self.qm.question(self,'CRISP-II',"Confrim exit CRISP ?\n Warning: Please save all necessary data" , self.qm.Yes | self.qm.No) == self.qm.No:
            return
        self.logger("Exiting CRISP...")
        sys.exit(app.exec_())
    #==========================[MAIN CODE LAUNCHER]=====================

    def cmd_opts(self):

    	if args.config_run == True:
    		gan.logger("[%s] | Running on GUI mode  : With commandline configuration"%time_stamp() , color = 3)
    	else:
    		gan.logger("[%s] | Running on GUI mode  : Normal"%time_stamp() , color = 3 )

    	# GUI style info
    	gan.logger("[%s] | GUI style mode       : %s"%( time_stamp(), qtStyle[args.gui_type]) )

    	if args.config_run == True:
    		gan.logger("[%s] | Running config file  : %s"%(time_stamp(),args.config_fpath) )
    		gan.load_config(config_fname = args.config_fpath,  msg = "")


    	# Run the configuration session directly from command line
    	if args.run_session == None:
    		return

    	elif args.run_session == "gan_train":
    		gan.init_logger()
    		gan.run_mode = "train_gan"
    		gan.update_vars()
    		hide_tf_warnings(supress_msg = gan.ignore_cls_warnings)
    		UseProcessor(gan.cls_process_type, mem_growth_flag = gan.gan_gpu_mem_growth)
    		gan.show_info()
    		gan.set_gan_plots()
    		gan.logger ("\n[%s] Press [u]: Update preview | [s]: Save model | [q]: Save & quit training on CRISP Preview window \n"%time_stamp() , color = 4)
    		gan.train_gan()

    	elif args.run_session == "gan_syn":
        	self.init_logger()
        	self.run_mode ="synthesis"
        	hide_tf_warnings(supress_msg = self.warn_status)                  # Hide Tensorflow warning messages to make consol output clear
        	UseProcessor(self.cls_process_type)
        	self.update_vars()                                                # get the latest updated variables for traning
        	self.show_info()
        	self.set_gan_plots()                                              # set gan plots
        	self.contours_synthsizer()

    	elif args.run_session == "cls_inf":
        	gan.init_logger() 
        	gan.run_mode = "classifier_inference"
        	gan.update_vars()
        	hide_tf_warnings(supress_msg = gan.ignore_cls_warnings)
        	UseProcessor(gan.cls_process_type)
        	gan.show_info()
        	gan.classifier()  

    	elif args.run_session == "cls_train":
        	gan.run_mode = "classifier_training"
        	gan.init_logger() 
        	gan.update_vars()
        	hide_tf_warnings(supress_msg = gan.ignore_cls_warnings)
        	UseProcessor(gan.cls_process_type)
        	gan.show_info()
        	gan.train_classifier()

    
 
if __name__=="__main__":

    init(autoreset=True)
    print(Style.BRIGHT + Fore.BLUE + show_logo())
    print(Style.BRIGHT + Fore.YELLOW + "Part of code associated with | CRISP II: Leveraging Deep Learning for Multigroup Classification in GC×GC-TOFMS of End-Stage Kidney Disease Patients")
    print(Style.BRIGHT + Fore.GREEN  + "Mathema et al 2025, Siriraj Metabolomics & Phenomics Center (SiMPC), Mahidol University, Bangkok,Thailand")
    print(Style.BRIGHT + Fore.GREEN  + "\nRunning on:[python %s]"%str(sys.version) ,  "\n"+ 60 *"=" ,"\n")

    if True:
        app=QApplication(sys.argv)
        app.setStyle(qtStyle[args.gui_type])
        gan=Model_GAN()
        gan.setWindowTitle('CRISP II: A deep learning approach for multiclass GC×GC-TOFMS contours separation of patients with end state renal disease')
        gan.show()
        gan.cmd_opts()

    sys.exit(app.exec_())         











