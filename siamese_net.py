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



# Siamese network for contour varification
import sys
import numpy as np
import os, warnings
import matplotlib.pyplot as plt
import cv2
#============================================== Keras and tensorflow
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input,  Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras.regularizers import l2
#=================================================

def hide_tf_warnings():                                                          
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

def euclidean_distance(vects):
    '''Compute Euclidean Distance between two vectors'''
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def create_base_network_signet(input_shape):          
    K.set_image_data_format('channels_last')

    seq = Sequential()
    seq.add(Conv2D(96, kernel_size=(11, 11), activation='relu', name='conv1_1', strides=4, input_shape= input_shape,init='glorot_uniform', dim_ordering='tf'))
    seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=3, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))    
    seq.add(ZeroPadding2D((2, 2), dim_ordering='tf'))
    
    seq.add(Conv2D(256, kernel_size=(5, 5), activation='relu', name='conv2_1', strides=1, init='glorot_uniform',  dim_ordering='tf'))
   
    seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=2, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    
    seq.add(Conv2D(384, kernel_size=(3, 3), activation='relu', name='conv3_1', strides=1, init='glorot_uniform',  dim_ordering='tf'))
    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    
    seq.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name='conv3_2', strides=1, init='glorot_uniform', dim_ordering='tf'))    
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(1024, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform'))
    seq.add(Dropout(0.5))    
    seq.add(Dense(128, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform')) # softmax changed to relu
    
    return seq

def siamese_model():
    # All the images will be converted to the same size before processing
    warnings.filterwarnings("ignore")                   # I got alot warnings due to version issues so ignore warnings here...
    img_h, img_w = 155, 220
    input_shape  =(img_h, img_w, 1)                     # input ut shape for base network

    # network definition
    base_network = create_base_network_signet(input_shape)

    input_a = Input(shape=(input_shape))
    input_b = Input(shape=(input_shape))
    # because we re-use the same instance `base_network`, the weights of the network will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Compute the Euclidean distance between the two vectors in the latent space
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)
    model.load_weights('./base_weights/siamese_net.h5')
    print ("\nSiamese base weights loaded...")

    return model 


def siamese_score(model, imgA, imgB):                                        # Predict distance score and classify test images as Genuine or Forged
    hide_tf_warnings()
    img_h, img_w = 155, 220                                                  # fixed input shape in siamese model (based on signature database)    
    #K.set_image_data_format('channels_last')                                # uncomment only if channeling causes problem
    pairs = [np.zeros((1, img_h, img_w, 1)) for i in range(2)]               # batch size
    img1 = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY) 
    img2 = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    img1=  cv2.resize(img1, (img_w, img_h))
    img2=  cv2.resize(img2, (img_w, img_h))
    img1 = np.array(img1, dtype = np.float64)/255
    img2 = np.array(img2, dtype = np.float64)/255
    img1 = img1[...,np.newaxis]
    img2 = img2[...,np.newaxis]
    pairs [0][0, :, :, :] = img1
    pairs [1][0, :, :, :] = img2
    img1=pairs[0]
    img2=pairs[1]
    test_label =1

    result = model.predict([img1, img2])
    threshold = 0.025
    diff = round( result[0][0], 5)
    #print("Difference Score = ", diff)
    #print("-->Its a Forged image") if  diff > threshold else None

    return diff
