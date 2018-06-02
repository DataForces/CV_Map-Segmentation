#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 22:33:02 2017

@author: kaku
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:39:59 2017

@author: kaku
"""
import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np

from bokeh.plotting import output_file, show, figure,vplot

import cv2, glob, random, os, datetime
from keras.models import Model
from keras.layers import Input, merge, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json

smooth = 1e-12

def image_and_name_read(path, form, name_read = False):
    """
    Read all the images and images names in path with a format.
    Return image and name list
    
    Parameters
    ----------
    path : str
        image file path
    form : str
        image format such as '*.tif'
    name_read : bool
        True get both image and name
        False only image 
    
    Returns
    -------
    image_list : list 
        list of all images with specific format in path
    name_list : list
        corresponding names
    """
    files = sorted(glob.glob(path + form))
    image_list = []
    [image_list.append(tiff.imread(image)) for image in files]
    if name_read == False:
        return image_list
    else:
        name_list = [os.path.basename(name) for name in files]
        return image_list, name_list

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
    
    x, y = np.zeros((patch_num,is2,is2,5)), np.zeros((patch_num,is2,is2,11))

    for i in range(patch_num):
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)

        im = img[xc:xc + is2, yc:yc + is2]
        ms = msk[xc:xc + is2, yc:yc + is2]

        if aug:
            if random.uniform(0, 1) > 0.5:
                im = im[::-1]
                ms = ms[::-1]

        ms = np_utils.to_categorical(ms, 11).reshape(patch_shape, -1, 11)

        x[i] = im
        y[i] = ms

    print (x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y))
    return x, y

def get_unet(patch_size):
    """
    Build a mini U-Net architecture
    Return U-Net model
    
    Notes
    -----
    Shape of output image is similar with input image
    Output img bands: N_Cls
    Upsampling is important
#    """
    ISZ = patch_size
    N_Cls = 11    
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
#    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
#    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model  

def save_model(model, model_path, file_time_global):
    """
    Save model into model_path
    """
    json_string=model.to_json()
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    modle_path=os.path.join(model_path,'architecture'+'_'+file_time_global+'.json')
    open(modle_path,'w').write(json_string)
    model.save_weights(os.path.join(model_path,'model_weights'+'_'+ file_time_global+'.h5'),overwrite= True )
    
def read_model(model_path, file_time_global):
    """
    Read model from model_path
    """
    model=model_from_json(open(os.path.join(model_path,'architecture'+'_'+ file_time_global+'.json')).read())
    model.load_weights(os.path.join(model_path,'model_weights'+'_'+file_time_global+'.h5'))
    return model

def acc_loss_visual(history, result_path, script_name, file_time_global):
    """
    Plot accuracy and loss 
    """
    acc_arr=np.array(history['acc'],dtype=float)
    loss_arr=np.array(history['loss'],dtype=float)
    val_acc_arr=np.array(history['val_acc'],dtype=float)
    val_loss_arr=np.array(history['val_loss'],dtype=float)
    output_file(os.path.join(result_path,"legends_"+script_name+"_"+file_time_global+".html"))
    p1 = figure()
    p1.title = "Training and Cross validation accuracy"
    p1.xaxis.axis_label = 'Iterations'
    p1.yaxis.axis_label = 'Accuracy'
    p1.circle(range(0,len(acc_arr)), acc_arr,color="green", legend="Train_Acc")
    p1.line(range(0,len(acc_arr)), acc_arr,line_color="green", legend="Train_Acc")
    p1.circle(range(0,len(val_acc_arr)), val_acc_arr,color="orange", legend="Val_Acc")
    p1.line(range(0,len(val_acc_arr)), val_acc_arr, line_color="orange",legend="Val_Acc")
    p2 = figure()
    p2.title = "Training and Cross validation loss_error"
    p2.xaxis.axis_label = 'Iterations'
    p2.yaxis.axis_label = 'Loss_error'
    p2.circle(range(0,len(loss_arr)), loss_arr,color="green", legend="Train_loss")
    p2.line(range(0,len(loss_arr)), loss_arr,line_color="green", legend="Train_loss")
    p2.circle(range(0,len(val_loss_arr)), val_loss_arr,color="orange", legend="Val_loss")
    p2.line(range(0,len(val_loss_arr)), val_loss_arr, line_color="orange",legend="Val_loss")
    p = vplot(p1, p2)
    show(p)

def image_test(model, image, patch_size):
    """
    Test input image,
    split image into small size with size patch_size,
    test splited images respectively.
    Returns result list
    
    Parameters
    ----------
        model : trained model
        image : np.array
            tested image
        patch_size : int
    Returns
    -------
        results : list
            test_y list 
    """
    test_X = image_split(image, patch_size)
    test_y = np.round(model.predict(test_X))
    results = []
    for i in range(test_X.shape[0]):
        result_small = test_y[i,:,:,:]
        real_result = np.argmax(result_small, axis = -1)
        results.append(real_result)
    return results

def image_split(image, patch_size):
    """
    Split a image into small size with size equles to patch_size,
    the boundary ones also choose size patch_size,
    some part may be overlapped with penultimate patch
    Return 4 dim images array
    
    Parameters
    ----------
    image : np.array
        input image
    patch_size : int
        
    Returns
    -------
    small_images : np.array
        4 dim image array, contains many split images
        
    Examples
    --------
        input: (image, 32)
            image size = 92, 113
        output: (12, 32, 32, 5)
            12 pieces of splited images with size 32 x 32 x 5
    """
    row_num = round(image.shape[0] / patch_size)
    col_num = round(image.shape[1] / patch_size)
    small_images = np.zeros((row_num * col_num, patch_size, patch_size, image.shape[-1]))
    row_max, col_max = row_num - 1, col_num - 1
    for row in range(row_num):
        for col in range(col_num):
            if row != row_max and col != col_max:
                small_image = image[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size, :]
            else:
                if row == row_max and col == col_max:
                    small_image = image[-patch_size:, -patch_size: ,:]
                else:
                    if col == col_max and row != row_max:
                        small_image = image[row*patch_size:(row+1)*patch_size, -patch_size:, :]
                    else:
                        small_image = image[-patch_size:, col*patch_size:(col+1)*patch_size, :]
            small_images[row * col_num + col, :,:,:] = small_image
    return small_images

def image_combine(image, small_images, patch_size):
    """
    Combine small images into a big one
    boundary image only choose part value
    Returns big combined image
    
    Parameters
    ----------
    image : np.array
        used to get combined image shape
    small_images : list 
        splited small images list
    patch_size
    
    Returns
    -------
    combined_image : np.array
        output combined image
    """
    row_num = round(image.shape[0] / patch_size)
    col_num = round(image.shape[1] / patch_size)
    combined_image = np.zeros((image.shape[0], image.shape[1]))
    row_max, col_max = row_num - 1, col_num - 1
    for row in range(row_num):
        for col in range(col_num):
            idx = row * col_num + col
            if row != row_max and col != col_max:
                combined_image[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] \
                = small_images[idx]
            else:
                if row == row_max and col == col_max:
                    combined_image[row*patch_size:, col*patch_size:] = \
                    small_images[idx][-(image.shape[0] - row*patch_size):, -(image.shape[1] - col*patch_size):]
                else:
                    if col == col_max and row != row_max:
                        combined_image[row*patch_size:(row+1)*patch_size, -(image.shape[1] - col*patch_size) :] = \
                        small_images[idx][:, -(image.shape[1] - col*patch_size) :]
                    else:
                        combined_image[-(image.shape[0] - row*patch_size) :, col*patch_size:(col+1)*patch_size] = \
                        small_images[idx][-(image.shape[0] - row*patch_size):, : ]
    combined_image = combined_image.astype(np.uint8)
    return combined_image

def close_processing(image):
    """
    Closing image processing
    Dilation followed by Erosion
    Kernel: 3 x 3
    """
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing

def post_processing(image):
    """
    Post processing
    Such as colse processing or others
    Returns processed image result 
    """
    post_result = close_processing(image)
    return post_result

def result_plot_and_save(ori_image, ori_mask, ori_image_name, model_name, result, result_path): 
    """
    Plot and save result image
    
    Parameters
    ----------
    ori_image : np.array
        original image
    name : str
        original image's name
    result : np.array
        result image
    result_path : str
    """
    result = close_processing(result)
    fig = plt.figure()
    plt.suptitle('Image ' + ori_image_name + ' segmentation result', y = 0.75, fontsize=11)
    plt.subplot(1,3,1)
    plt.xlabel('ori_image')
    plt.imshow(ori_image[:,:,:3], vmin=0, vmax=11)
    
    plt.subplot(1,3,2)
    plt.xlabel('seg_result')
    plt.imshow(result, vmin=0, vmax=11)

    plt.subplot(1,3,3)
    plt.xlabel('ground_truth')
    plt.imshow(ori_mask, vmin=0, vmax=11)

    fig.savefig(os.path.join(result_path, ori_image_name + '_'+ model_name +'_compared_result.png'))
    plt.imsave(os.path.join(result_path, ori_image_name + '_' + model_name +'_result.png'), result, vmin=0, vmax=11)
    plt.show()
    
if __name__ == '__main__':
    ###########  file colation and globle V  ###########
    now_global = datetime.datetime.now()
    time_global = str(now_global.strftime("%Y-%m-%d-%H-%M")) 
    script_name = os.path.basename(__file__)
    
    raster_path = '../Data/UseZone_raster4326/'
    vector_path = '../Data/UseZone_vector4326/'
    model_path = '../Model/'
    result_path = '../Result/'
    model_name = '2017-05-18-13-57'
    
    model_train = True
    
    PATCH_SIZE = 32
    BATCH_SIZE = 32
    EPOCH = 10
    TRINING_SAMPLES = 10000
    
    ##############################  data preparing ###########################
    raster_images, raster_images_name = image_and_name_read(raster_path, '*.tif', name_read = True)
    vector_images = resize_vector_image(image_and_name_read(vector_path, '*.tif'), raster_images[0])
    
    ##############################  model training ###########################
    if model_train == True:
        model_name = time_global
        stacked_raster, stacked_vector = stick_all_train(raster_images, vector_images)
        train_X, train_y = get_patches(stacked_raster, stacked_vector, aug=True, patch_num=TRINING_SAMPLES, patch_size=PATCH_SIZE)        
        model = get_unet(PATCH_SIZE)
        History = model.fit(x = train_X, y = train_y, batch_size=BATCH_SIZE, nb_epoch = EPOCH, verbose=1, validation_split=0.2)
        save_model(model, model_path, time_global)
        acc_loss_visual(History.history, result_path, script_name, time_global)
    else:
        model = read_model(model_path, model_name)
        
    ############################## image testing ###########################
    for i in range(len(raster_images)-1):
        test_image = raster_images[i]
        test_y = image_test(model, test_image, PATCH_SIZE)   
    
    ############################## result ##################################
        combined_image = image_combine(test_image, test_y, PATCH_SIZE)
        
        seg_result = post_processing(combined_image)

        result_plot_and_save(test_image, vector_images[i], raster_images_name[i], model_name, seg_result, result_path)
    
    
    
    
    
    