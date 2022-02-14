# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 12:23:08 2021

@author: aya
"""

import glob
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import unsharp_mask
from skimage.feature import greycomatrix, greycoprops
from skimage import color, img_as_ubyte
import math
from sklearn.utils import shuffle
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import itertools
from sklearn.metrics import accuracy_score


# path
path_colon_aca = glob.glob("D:/Stage/colon_image_sets/colon_aca/*.jpeg")
path_colon_n = glob.glob("D:/Stage/colon_image_sets/colon_n/*.jpeg")
path_lung_aca = glob.glob("D:/Stage/lung_image_sets/lung_aca/*.jpeg")
path_lung_n = glob.glob("D:/Stage/lung_image_sets/lung_n/*.jpeg")
path_lung_scc = glob.glob("D:/Stage/lung_image_sets/lung_scc/*.jpeg")


def read_images(path):
    images = []
    i = 0
    for img in path:
        n = cv.imread(img)
        img = cv.resize(n,(200,200))
        #plt.imshow(img)
        #plt.show()
        images.append(img)
        i += 1
        if i == 500:
            break   
    return(images)

img_colon_aca = read_images(path_colon_aca)
img_colon_n = read_images(path_colon_n)
img_lung_aca = read_images(path_lung_aca)
img_lung_n = read_images(path_lung_n)
img_lung_scc = read_images(path_lung_scc)


# Unsharp Masking
radius = 2
amount = 5

def masque_flou (images):
    img_Masque_Flou = []
    for imgMasque in images:
        result = unsharp_mask(imgMasque, radius=radius, amount=amount)
        #plt.imshow(result, cmap=plt.cm.gray)
        #plt.show()
        img_Masque_Flou.append(result)
    return(img_Masque_Flou)

img_Masque_colon_aca = masque_flou(img_colon_aca)
img_Masque_colon_n = masque_flou(img_colon_n) 
img_Masque_lung_aca = masque_flou(img_lung_aca) 
img_Masque_lung_n = masque_flou(img_lung_n) 
img_Masque_lung_scc = masque_flou(img_lung_scc)  


# First order statistics 
def mean(img_Masque_Flou):
    Mean = []    
    for imgMasque in img_Masque_Flou:
        mean = np.mean(imgMasque)
        Mean.append(mean)
    return(Mean)

def std(img_Masque_Flou):   
    Std = []    
    for imgMasque in img_Masque_Flou:
        std = np.std(imgMasque)
        Std.append(std)
    return(Std)

def median(img_Masque_Flou):   
    Median = []    
    for imgMasque in img_Masque_Flou:
        med = np.median(imgMasque)
        Median.append(med)
    return(Median)

def quantile(img_Masque_Flou): 
    Quantile = []    
    for imgMasque in img_Masque_Flou:
        quantile = np.quantile(imgMasque, [0.25, 0.5, 0.75])
        Quantile.append(quantile)
    return(Quantile)


def data_mean(Mean):
    data_Mean = pd.DataFrame(Mean)
    data_Mean.columns = ['Mean']
    return(data_Mean)

def data_std(Std):
    data_std = pd.DataFrame(Std)
    data_std.columns = ['Std']
    return(data_std)

def data_median(Median):
    data_med = pd.DataFrame(Median)
    data_med.columns = ['Median']
    return(data_med)

def data_quantile(Quantile):
    data_quantile = pd.DataFrame(Quantile)
    data_quantile.columns = ['Percentile 25%', 'Percentile 50%', 'Percentile 75%']
    return(data_quantile)


# colon_aca       
Mean_colon_aca = mean(img_Masque_colon_aca)   
data_Mean_colon_aca = data_mean(Mean_colon_aca)

std_colon_aca = std(img_Masque_colon_aca)   
data_std_colon_aca = data_std(std_colon_aca)

Med_colon_aca = median(img_Masque_colon_aca)   
data_Med_colon_aca = data_median(Med_colon_aca)
    
quantile_colon_aca = quantile(img_Masque_colon_aca)     
data_quantile_colon_aca = data_quantile(quantile_colon_aca)


# colon_n
Mean_colon_n = mean(img_Masque_colon_n)   
data_Mean_colon_n = data_mean(Mean_colon_n)

std_colon_n = std(img_Masque_colon_n)   
data_std_colon_n = data_std(std_colon_n)

Med_colon_n = median(img_Masque_colon_n)   
data_Med_colon_n = data_median(Med_colon_n)
    
quantile_colon_n = quantile(img_Masque_colon_n)     
data_quantile_colon_n = data_quantile(quantile_colon_n)

# lung_aca
Mean_lung_aca = mean(img_Masque_lung_aca)   
data_Mean_lung_aca = data_mean(Mean_lung_aca)

std_lung_aca = std(img_Masque_lung_aca)   
data_std_lung_aca = data_std(std_lung_aca)

Med_lung_aca = median(img_Masque_lung_aca)   
data_Med_lung_aca = data_median(Med_lung_aca)
    
quantile_lung_aca = quantile(img_Masque_lung_aca)     
data_quantile_lung_aca = data_quantile(quantile_lung_aca)

# lung_n
Mean_lung_n = mean(img_Masque_lung_n)   
data_Mean_lung_n = data_mean(Mean_lung_n)

std_lung_n = std(img_Masque_lung_n)   
data_std_lung_n = data_std(std_lung_n)

Med_lung_n = median(img_Masque_lung_n)   
data_Med_lung_n = data_median(Med_lung_n)
    
quantile_lung_n = quantile(img_Masque_lung_n)     
data_quantile_lung_n = data_quantile(quantile_lung_n)

# lung_scc
Mean_lung_scc = mean(img_Masque_lung_scc)   
data_Mean_lung_scc = data_mean(Mean_lung_scc)

std_lung_scc = std(img_Masque_lung_scc)   
data_std_lung_scc = data_std(std_lung_scc)

Med_lung_scc = median(img_Masque_lung_scc)   
data_Med_lung_scc = data_median(Med_lung_scc)
    
quantile_lung_scc = quantile(img_Masque_lung_scc)     
data_quantile_lung_scc = data_quantile(quantile_lung_scc)


# GLCM properties
def contrast_feature(matrix_coocurrence):
	contrast = greycoprops(matrix_coocurrence, 'contrast')
	return contrast

def dissimilarity_feature(matrix_coocurrence):
	dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')	
	return dissimilarity

def homogeneity_feature(matrix_coocurrence):
	homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
	return homogeneity

def energy_feature(matrix_coocurrence):
	energy = greycoprops(matrix_coocurrence, 'energy')
	return energy

def correlation_feature(matrix_coocurrence):
	correlation = greycoprops(matrix_coocurrence, 'correlation')
	return correlation

def asm_feature(matrix_coocurrence):
	asm = greycoprops(matrix_coocurrence, 'ASM')
	return asm

bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit

def matrice_co_occureence(img_Masque_Flou):
    matrix_coocurrence = []
    for Masque in img_Masque_Flou:
        gray = color.rgb2gray(Masque)
        image = img_as_ubyte(gray)
        inds = np.digitize(image, bins)
        max_value = inds.max()+1
        matrixcoocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
        matrix_coocurrence.append(matrixcoocurrence)
    return(matrix_coocurrence)


def img_gray(img_Masque_Flou):
    imggray = []
    for Masque in img_Masque_Flou:
        gray = color.rgb2gray(Masque)
        imggray.append(gray)
    return(imggray)


# colon_aca    
matrix_coocurrence_colon_aca =  matrice_co_occureence(img_Masque_colon_aca)  
imggray_colon_aca = img_gray(img_Masque_colon_aca)

# colon_n
matrix_coocurrence_colon_n =  matrice_co_occureence(img_Masque_colon_n)  
imggray_colon_n = img_gray(img_Masque_colon_n)

# lung_aca
matrix_coocurrence_lung_aca = matrice_co_occureence(img_Masque_lung_aca)
imggray_lung_aca = img_gray(img_Masque_lung_aca)

# lung_n
matrix_coocurrence_lung_n = matrice_co_occureence(img_Masque_lung_n)
imggray_lung_n = img_gray(img_Masque_lung_n)

# lung_scc
matrix_coocurrence_lung_scc = matrice_co_occureence(img_Masque_lung_scc)
imggray_lung_scc = img_gray(img_Masque_lung_scc)


def contrast(matrix_coocurrence):
    Contrast_feature = []
    for matrix_co in matrix_coocurrence:
        Contrast_feature.append(contrast_feature(matrix_co))  
    return(Contrast_feature)

def dissimilarity(matrix_coocurrence):
    Dissimilarity_feature = []
    for matrix_co in matrix_coocurrence:
        Dissimilarity_feature.append(dissimilarity_feature(matrix_co))  
    return(Dissimilarity_feature)

def homogeneity(matrix_coocurrence):
    Homogeneity_feature = []
    for matrix_co in matrix_coocurrence:
        Homogeneity_feature.append(homogeneity_feature(matrix_co))  
    return(Homogeneity_feature)

def energy(matrix_coocurrence):
    Energy_feature = []
    for matrix_co in matrix_coocurrence:
        Energy_feature.append(energy_feature(matrix_co))  
    return(Energy_feature)

def correlation(matrix_coocurrence):
    Correlation_feature = []
    for matrix_co in matrix_coocurrence:
        Correlation_feature.append(correlation_feature(matrix_co))  
    return(Correlation_feature)

def asm(matrix_coocurrence):
    Asm_feature = []
    for matrix_co in matrix_coocurrence:
        Asm_feature.append(asm_feature(matrix_co))  
    return(Asm_feature)


def data_Contrast(Contrast_feature):
    contrast_feature_np = np.array(Contrast_feature)
    contrast_feature = contrast_feature_np.reshape(-1)  
    contrast_feature = contrast_feature.reshape(len(contrast_feature_np), 4)  
    data_contrast = pd.DataFrame(contrast_feature)
    data_contrast.columns = ['Contrast1', 'Contrast2', 'Contrast3', 'Contrast4']
    return(data_contrast)

def data_Dissimilarity(Dissimilarity_feature):
    dissimilarity_feature_np = np.array(Dissimilarity_feature)
    dissimilarity_feature = dissimilarity_feature_np.reshape(-1)
    dissimilarity_feature = dissimilarity_feature.reshape(len(dissimilarity_feature_np), 4)  
    data_dissimilarity = pd.DataFrame(dissimilarity_feature)
    data_dissimilarity.columns = ['dissimilarity1', 'dissimilarity2', 'dissimilarity3', 'dissimilarity4']
    return(data_dissimilarity)

def data_Homogeneity(Homogeneity_feature):
    homogeneity_feature_np = np.array(Homogeneity_feature)
    homogeneity_feature = homogeneity_feature_np.reshape(-1)  
    homogeneity_feature = homogeneity_feature.reshape(len(homogeneity_feature_np), 4)  
    data_homogeneity = pd.DataFrame(homogeneity_feature)
    data_homogeneity.columns = ['homogeneity1', 'homogeneity2', 'homogeneity3', 'homogeneity4']
    return(data_homogeneity)

def data_Energy(Energy_feature):
    energy_feature_np = np.array(Energy_feature)
    energy_feature = energy_feature_np.reshape(-1)
    energy_feature = energy_feature.reshape(len(energy_feature_np), 4)  
    data_energy = pd.DataFrame(energy_feature)
    data_energy.columns = ['energy1', 'energy2', 'energy3', 'energy4']
    return(data_energy)

def data_Correlation(Correlation_feature):
    correlation_feature_np = np.array(Correlation_feature)
    correlation_feature = correlation_feature_np.reshape(-1) 
    correlation_feature = correlation_feature.reshape(len(correlation_feature_np), 4)  
    data_correlation = pd.DataFrame(correlation_feature)
    data_correlation.columns = ['correlation1', 'correlation2', 'correlation3', 'correlation4']
    return(data_correlation)
    
def data_Asm(Asm_feature):    
    asm_feature_np = np.array(Asm_feature)
    asm_feature = asm_feature_np.reshape(-1)
    asm_feature = asm_feature.reshape(len(asm_feature_np), 4)  
    data_asm = pd.DataFrame(asm_feature)
    data_asm.columns = ['asm1', 'asm2', 'asm3', 'asm4']
    return(data_asm)
    
# colon_aca    
contrast_feature_colon_aca = contrast(matrix_coocurrence_colon_aca)
dissimilarity_feature_colon_aca = dissimilarity(matrix_coocurrence_colon_aca)
homogeneity_feature_colon_aca = homogeneity(matrix_coocurrence_colon_aca)
energy_feature_colon_aca = energy(matrix_coocurrence_colon_aca)
correlation_feature_colon_aca = correlation(matrix_coocurrence_colon_aca)
asm_feature_colon_aca = asm(matrix_coocurrence_colon_aca)
        
data1_contrast_colon_aca = data_Contrast(contrast_feature_colon_aca)
data1_dissimilarity_colon_aca = data_Dissimilarity(dissimilarity_feature_colon_aca)  
data1_homogeneity_colon_aca = data_Homogeneity(homogeneity_feature_colon_aca)
data1_energy_colon_aca = data_Energy(energy_feature_colon_aca)   
data1_correlation_colon_aca = data_Correlation(correlation_feature_colon_aca)  
data1_asm_colon_aca = data_Asm(asm_feature_colon_aca)   

# colon_n
contrast_feature_colon_n = contrast(matrix_coocurrence_colon_n)
dissimilarity_feature_colon_n = dissimilarity(matrix_coocurrence_colon_n)
homogeneity_feature_colon_n = homogeneity(matrix_coocurrence_colon_n)
energy_feature_colon_n = energy(matrix_coocurrence_colon_n)
correlation_feature_colon_n = correlation(matrix_coocurrence_colon_n)
asm_feature_colon_n = asm(matrix_coocurrence_colon_n)
        
data2_contrast_colon_n = data_Contrast(contrast_feature_colon_n)
data2_dissimilarity_colon_n = data_Dissimilarity(dissimilarity_feature_colon_n)  
data2_homogeneity_colon_n = data_Homogeneity(homogeneity_feature_colon_n)
data2_energy_colon_n = data_Energy(energy_feature_colon_n)   
data2_correlation_colon_n = data_Correlation(correlation_feature_colon_n)  
data2_asm_colon_n = data_Asm(asm_feature_colon_n)   

# lung_aca
contrast_feature_lung_aca = contrast(matrix_coocurrence_lung_aca)
dissimilarity_feature_lung_aca = dissimilarity(matrix_coocurrence_lung_aca)
homogeneity_feature_lung_aca = homogeneity(matrix_coocurrence_lung_aca)
energy_feature_lung_aca = energy(matrix_coocurrence_lung_aca)
correlation_feature_lung_aca = correlation(matrix_coocurrence_lung_aca)
asm_feature_lung_aca = asm(matrix_coocurrence_lung_aca)

data3_contrast_lung_aca = data_Contrast(contrast_feature_lung_aca)
data3_dissimilarity_lung_aca = data_Dissimilarity(dissimilarity_feature_lung_aca)  
data3_homogeneity_lung_aca = data_Homogeneity(homogeneity_feature_lung_aca)
data3_energy_lung_aca = data_Energy(energy_feature_lung_aca)   
data3_correlation_lung_aca = data_Correlation(correlation_feature_lung_aca)  
data3_asm_lung_aca = data_Asm(asm_feature_lung_aca)   

# lung_n
contrast_feature_lung_n = contrast(matrix_coocurrence_lung_n)
dissimilarity_feature_lung_n = dissimilarity(matrix_coocurrence_lung_n)
homogeneity_feature_lung_n = homogeneity(matrix_coocurrence_lung_n)
energy_feature_lung_n = energy(matrix_coocurrence_lung_n)
correlation_feature_lung_n = correlation(matrix_coocurrence_lung_n)
asm_feature_lung_n = asm(matrix_coocurrence_lung_n)

data4_contrast_lung_n = data_Contrast(contrast_feature_lung_n)
data4_dissimilarity_lung_n = data_Dissimilarity(dissimilarity_feature_lung_n)  
data4_homogeneity_lung_n = data_Homogeneity(homogeneity_feature_lung_n)
data4_energy_lung_n = data_Energy(energy_feature_lung_n)   
data4_correlation_lung_n = data_Correlation(correlation_feature_lung_n)  
data4_asm_lung_n = data_Asm(asm_feature_lung_n)

# lung_scc
contrast_feature_lung_scc = contrast(matrix_coocurrence_lung_scc)
dissimilarity_feature_lung_scc = dissimilarity(matrix_coocurrence_lung_scc)
homogeneity_feature_lung_scc = homogeneity(matrix_coocurrence_lung_scc)
energy_feature_lung_scc = energy(matrix_coocurrence_lung_scc)
correlation_feature_lung_scc = correlation(matrix_coocurrence_lung_scc)
asm_feature_lung_scc = asm(matrix_coocurrence_lung_scc)

data5_contrast_lung_scc = data_Contrast(contrast_feature_lung_scc)
data5_dissimilarity_lung_scc = data_Dissimilarity(dissimilarity_feature_lung_scc)  
data5_homogeneity_lung_scc = data_Homogeneity(homogeneity_feature_lung_scc)
data5_energy_lung_scc = data_Energy(energy_feature_lung_scc)   
data5_correlation_lung_scc = data_Correlation(correlation_feature_lung_scc)  
data5_asm_lung_scc = data_Asm(asm_feature_lung_scc) 
   

# Calculate Hu Moments Invariants
def Moments(imggray):
    hu_Moments = []
    for Masque in imggray:
        moments = cv.moments(Masque)
        huMoments = cv.HuMoments(moments)
        for i in range(0,7):
            huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))
            hu_Moments.append(huMoments[i])
    return(hu_Moments)       

def data_Moments(hu_Moments):
    Moments_np = np.array(hu_Moments)
    hu_Moments = Moments_np.reshape(-1) 
    hu_Moments = hu_Moments.reshape(len(img_colon_aca), 7)  
    data_Moments = pd.DataFrame(hu_Moments)
    data_Moments.columns = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7']
    return(data_Moments)

# colon_aca
hu_Moments_colon_aca = Moments(imggray_colon_aca)
data1_Moments_colon_aca = data_Moments(hu_Moments_colon_aca)

data1_colon_aca = pd.concat([data_Mean_colon_aca, data_std_colon_aca, data_Med_colon_aca, data_quantile_colon_aca, data1_contrast_colon_aca, data1_dissimilarity_colon_aca, data1_homogeneity_colon_aca, data1_energy_colon_aca, data1_correlation_colon_aca, data1_asm_colon_aca, data1_Moments_colon_aca], axis=1)

colon_aca=["Colon_aca"]
for i in range(len(img_colon_aca) - 1):
     colon_aca[i+1]=colon_aca.append("")
     colon_aca[i+1]="Colon_aca"
     
data1_colon_aca['target'] = colon_aca


# colon_n
hu_Moments_colon_n = Moments(imggray_colon_n)
data2_Moments_colon_n = data_Moments(hu_Moments_colon_n)

data2_colon_n = pd.concat([data_Mean_colon_n, data_std_colon_n, data_Med_colon_n, data_quantile_colon_n, data2_contrast_colon_n, data2_dissimilarity_colon_n, data2_homogeneity_colon_n, data2_energy_colon_n, data2_correlation_colon_n, data2_asm_colon_n, data2_Moments_colon_n], axis=1)

colon_n=["Colon_n"]
for i in range(len(img_colon_n) - 1):
     colon_n[i+1]=colon_n.append("")
     colon_n[i+1]="Colon_n"
     
data2_colon_n['target'] = colon_n

data_colon = data1_colon_aca.append(data2_colon_n)


# lung_aca
hu_Moments_lung_aca = Moments(imggray_lung_aca)
data3_Moments_lung_aca = data_Moments(hu_Moments_lung_aca)

data3_lung_aca = pd.concat([data_Mean_lung_aca, data_std_lung_aca, data_Med_lung_aca, data_quantile_lung_aca, data3_contrast_lung_aca, data3_dissimilarity_lung_aca, data3_homogeneity_lung_aca, data3_energy_lung_aca, data3_correlation_lung_aca, data3_asm_lung_aca, data3_Moments_lung_aca], axis=1)

lung_aca = ["Lung_aca"]
for i in range(len(img_lung_aca) - 1):
     lung_aca[i+1] = lung_aca.append("")
     lung_aca[i+1] = "Lung_aca"
     
data3_lung_aca['target'] = lung_aca


# lung_n
hu_Moments_lung_n = Moments(imggray_lung_n)
data4_Moments_lung_n = data_Moments(hu_Moments_lung_n)

data4_lung_n = pd.concat([data_Mean_lung_n, data_std_lung_n,  data_Med_lung_n, data_quantile_lung_n,data4_contrast_lung_n, data4_dissimilarity_lung_n, data4_homogeneity_lung_n, data4_energy_lung_n, data4_correlation_lung_n, data4_asm_lung_n, data4_Moments_lung_n], axis=1)

lung_n = ["Lung_n"]
for i in range(len(img_lung_n) - 1):
     lung_n[i+1] = lung_n.append("")
     lung_n[i+1] = "Lung_n"
     
data4_lung_n['target'] = lung_n

data_lung_aca_n = data3_lung_aca.append(data4_lung_n)


# lung_scc
hu_Moments_lung_scc = Moments(imggray_lung_scc)
data5_Moments_lung_scc = data_Moments(hu_Moments_lung_scc)

data5_lung_scc = pd.concat([data_Mean_lung_scc, data_std_lung_scc,  data_Med_lung_scc, data_quantile_lung_scc, data5_contrast_lung_scc, data5_dissimilarity_lung_scc, data5_homogeneity_lung_scc, data5_energy_lung_scc, data5_correlation_lung_scc, data5_asm_lung_scc, data5_Moments_lung_scc], axis=1)

lung_scc = ["Lung_scc"]
for i in range(len(img_lung_scc) - 1):
     lung_scc[i+1] = lung_scc.append("")
     lung_scc[i+1] = "Lung_scc"
     
data5_lung_scc['target'] = lung_scc

data_lung = data_lung_aca_n.append(data5_lung_scc)


Data = data_colon.append(data_lung)

# ----------------------------------

Datas = shuffle(Data)

X = Datas.drop(['target'], axis = 1)
Y = Datas.target.values

# We will split our data. 80% of our data will be train data and 
# 20% of it will be test data.
X_train, X_test, Y_train_label, Y_test_label = train_test_split(X,Y,test_size = 0.2,random_state=0)

# check the shape of X_train and X_test
X_train.shape, X_test.shape

# Transforming non numerical labels into numerical labels
encoder = preprocessing.LabelEncoder()

# encoding train labels 
encoder.fit(Y_train_label)
Y_train = encoder.transform(Y_train_label)

# encoding test labels 
encoder.fit(Y_test_label)
Y_test = encoder.transform(Y_test_label)

# Scaling the Train and Test feature set 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Set the input shape
feature_vector_length = len(Datas.columns) - 1
input_shape =(feature_vector_length,)
num_classes = len(np.unique(Y_train))

# MLPClassifier
# Create the model
model = Sequential()
model.add(Dense(150, input_shape=input_shape, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# Configure the model and start training
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_scaled, Y_train, epochs=300, batch_size=200, validation_split=0.2)

# Predict the Test set results
Y_pred = model.predict(X_test_scaled)
Y_pred.shape
Y_pred_max = np.zeros((Y_test.shape[0]))

for i in range(Y_test.shape[0]):
    Y_pred_max[i] = np.argmax(Y_pred[i])
    
np.unique(Y_pred_max)

# Check accuracy score 
accuracy_test = accuracy_score(Y_test, Y_pred_max)
print('\naccuracy_score on test dataset : ', accuracy_test)

# Making the Confusion Matrix
target_names = np.unique(Y)
print(confusion_matrix(Y_test,Y_pred_max))
print("\n")
print(classification_report(Y_test, Y_pred_max, target_names=target_names))


cm = confusion_matrix(Y_test,Y_pred_max)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
class_names = ['Colon_Ad', 'Colon_Be', 'Lung_Ad', 'Lung_Be', 'Lung_Sc']
plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')

print('Confusion matrix Accuracy is: {}'.format(metrics.accuracy_score(Y_test,Y_pred_max)))

