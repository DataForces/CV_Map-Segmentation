#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:39:59 2017

@author: kaku
"""
import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np
#from PIL import Image
import cv2, glob, random
from keras.models import Model, load_model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils import np_utils

smooth = 1e-12

def image_read(path, form):
    """
    Read all the images in path with a format.
    Return image list.
    
    Parameters
    ----------
    path : str
        image file path
    form : str
        image format such as '*.tif'
    
    Returns
    -------
    image_list : list 
        list of all images with specific format in path
    """
    files = sorted(glob.glob(path + form))
    image_list = []
    [image_list.append(tiff.imread(image)) for image in files]
    return image_list

def resize_vector_image(vector_images, raster_image):
    """
    Resize vector_image size, and change into unit8
    make size equal with raster_image,
    Return resized_vecter_images
    
    Parameters
    ----------
    vector_images : list
    raster_image : np.array
        size template
        
    Returns
    -------
    resized_images : list
        resized vector images list with dtype np.unint8
    """
    cols_size = raster_image.shape[1]
    rows_size = raster_image.shape[0]
    resized_images = []
    for vector_image in vector_images:    
        resized_image = np.round(cv2.resize(vector_image,(cols_size, rows_size))).astype(np.uint8)
        resized_images.append(resized_image)
    return resized_images  

def stick_all_train(raster_images, vector_images):
    """
    Stake all the training images into a large one row_num x col_num.
    try to make shape like a square 
    raster_image contains 5 bands, vecter_image contains 1 band.
    
    Returns
    -------
    x_trn_N_Cls : npy file
        5 bands
    y_trn_N_Cls : npy file
        1 bands
    """
    print ("Let's stick all imgs together")
    rows = raster_images[0].shape[0] #
    cols = raster_images[0].shape[1]    
    num_bands = raster_images[0].shape[2]
    
    num = int(np.sqrt(len(raster_images))) # deside sticked size, shape like square
    if rows < cols:
        row_num = num + 1
        col_num = num 
    else:
        row_num = num
        col_num = num + 1
    if row_num * col_num > rows * cols:
        row_num = num
        col_num = num 
    
    x = np.zeros((row_num * rows, col_num * cols, num_bands)) # DF.ImageId.unique() == 25, make image 5 x 5 togeter
    y = np.zeros((row_num * rows, col_num * cols, 1))

    print('Total image picecs:{}'.format(len(raster_images)))
    for i in range(row_num):
        for j in range(col_num):
            x[rows * i:rows * (i + 1), cols * j: cols * (j + 1), :] = raster_images[i * (row_num - 1) + j]
            y[rows * i:rows * (i + 1), cols * j: cols * (j + 1), 0] = vector_images[i * (row_num - 1) + j]
#            print(i,j)
    print ('Ground Truth ranges from {} to {}; \nLabel ranges from {} to {}. \nSticked shape : {} x {} = {}\n '\
           .format(np.amin(x), np.amax(x), np.amin(y.astype(np.int8)), np.amax(y.astype(np.int8)), row_num, col_num, row_num * col_num))
    np.save('data/x_trn_%d' % num_bands, x)
    np.save('data/y_trn_%d' % 1, y)
#    return x.astype(np.int8), y.astype(np.int8)
    return x, y.astype(np.int8)

def get_patches(img, msk, aug=True, **kwargs):
    """   
    Get patches from big image, used to be samples.
    Return: image pathes with size patch_shape, and num patch_num
    
    Parameters
    ----------
    img : np.array
    msk : np.array
    **kwargs:
        patch_num : samples number 
        patch_shape : patch shape size
    aug = True
        data augmentation method, here like rotate or change patch's sequence
    
    Notes
    -----
    2 * np.transpose(x, (0, 3, 1, 2)) - 1: feature scaling  
    """
    patch_num = kwargs['patch_num']
    patch_shape = kwargs['patch_size']  # sample shape size
    is2 = int(1.0 * patch_shape)
    xm, ym = img.shape[0] - is2, img.shape[1] - is2
    
    x, y = np.zeros((patch_num,is2,is2,5)), np.zeros((patch_num,is2,is2,1))

    for i in range(patch_num):
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)

        im = img[xc:xc + is2, yc:yc + is2]
        ms = msk[xc:xc + is2, yc:yc + is2]

        if aug:
            if random.uniform(0, 1) > 0.5:
                im = im[::-1]
                ms = ms[::-1]

        x[i] = im
        y[i] = ms

    print (x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y))
    return x, y

def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def get_unet(patch_size):
    """
    Build a mini U-Net architecture
    Return U-Net model
    
    Notes
    -----
    Shape of output image is similar with input image
    Output img piece: N_Cls
    Upsampling importance
#    """
    ISZ = patch_size
    N_Cls = 1    
    inputs = Input((ISZ, ISZ, 5))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(N_Cls, (1, 1), activation='softmax')(conv9)
#    conv10 = Conv2D(N_Cls, (1, 1), activation='softmax')(conv9)
    model = Model(input=inputs, output=conv10)
#    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
#    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model  

if __name__ == '__main__':
    raster_path = '../Data/UseZone_raster4326/'
    vector_path = '../Data/UseZone_vector4326/'
    patch_size = 32
    
    
    raster_images = image_read(raster_path, '*.tif')
    vector_images = resize_vector_image(image_read(vector_path, '*.tif'), raster_images[0])
    stacked_raster, stacked_vector = stick_all_train(raster_images, vector_images)
    patch_X, patch_y = get_patches(stacked_raster, stacked_vector, aug=True, patch_num=10000, patch_size=patch_size)
#    patch_y = np_utils.to_categorical(patch_y, 11)
    model = get_unet(patch_size)
    model.fit(x = patch_X, y = patch_y, batch_size=32, nb_epoch = 10,verbose=1)
    
    
    
    
    
    
    
    