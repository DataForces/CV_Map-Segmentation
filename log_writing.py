#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 23:13:39 2017

@author: kaku
"""
import os, sys
import numpy as np

def log_write(result_path, time_global, script_name, vector_images, PATCH_SIZE, N_ClASS, TRINING_SAMPLES, BATCH_SIZE,\
              EPOCH, time_end_train, time_beg_train, test_images, time_end_test, time_beg_test, History, \
              inter_over_unions_list, CV_RATIO, test_images_name, model, test_mode = True):
    stdout = sys.stdout
    
    log_file=open(os.path.join(result_path,'my_log.txt'),'a')
    
    sys.stdout = log_file
    
    print('########################Time: '+time_global+'########################')
    print('############################File: '+script_name+'########################')
    print('Training sample size: '+''+str(PATCH_SIZE)+' x '+str(PATCH_SIZE))
    print('Number of class: '+str(N_ClASS))
    print('Number of trianing samples: '+str(TRINING_SAMPLES * (1-CV_RATIO)))
    print('          viladation samples: '+str(TRINING_SAMPLES * CV_RATIO))    
    print('Batch_size: '+str(BATCH_SIZE))
    print('Iteration: '+str(EPOCH))
    print('Training_time: '+str(time_end_train-time_beg_train)+'    Every_iter:'+str((time_end_train-time_beg_train)/EPOCH))
    print('Training:')
    print('         accuracy: ' + str(History.history['acc'][-1])+'     loss: '+str(History.history['loss'][-1]))
    print('         jaccard_coef: ' + str(History.history['jaccard_coef'][-1])+'     jaccard_coef_int: '+str(History.history['jaccard_coef_int'][-1]))    
    print('Validation:'+ '\n')
    print('         accuracy: ' + str(History.history['val_acc'][-1])+'     loss: '+str(History.history['val_loss'][-1]))
    print('         jaccard_coef: ' + str(History.history['val_jaccard_coef'][-1])+'     jaccard_coef_int: '+str(History.history['val_jaccard_coef_int'][-1]))
    print("\n")    
    if test_mode == True:
        print('Testing image pieces: '+str(len(test_images)))
        print('Testing image size: ' + str(vector_images[0].shape[0])+' x '+ str(vector_images[0].shape[1]))
        print('Testing_time: '+str(time_end_test-time_beg_test)+'    Every image:'+str((time_end_test-time_beg_test)/len(test_images)))
        print("Testing result:")
        print("         Mean Intersection over Union value: "+'%.2f'%(np.mean(inter_over_unions_list)*100)+r'%')        
        print("\n")

    print('Model structure:')
    print('Input tensor:')
    print('             X: (Batch_num,'+str(PATCH_SIZE)+','+ str(PATCH_SIZE)+','+ str(test_images[0].shape[-1])+')')
    print('             Y: (Batch_num,'+str(PATCH_SIZE)+','+ str(PATCH_SIZE)+','+ str(N_ClASS) +')')    

    
    print('1    Convolution2D: 32,3,3  same relu ')
    print('1.5  Convolution2D: 32,3,3  same relu ')
    print('2    MaxPooling2D: pool_size=(2, 2) ')

    print('3    Convolution2D: 64,3,3  same relu ')
    print('3.5  Convolution2D: 64,3,3  same relu ')
    print('4    MaxPooling2D: pool_size=(2, 2) ')

    print('5    Convolution2D: 128,3,3  same relu ')
    print('5.5  Convolution2D: 128,3,3  same relu ')
    print('6    MaxPooling2D: pool_size=(2, 2) ')

    print('7    Convolution2D: 128,3,3  same relu ')
    print('7.5  Convolution2D: 128,3,3  same relu ')
    print('8    MaxPooling2D: pool_size=(2, 2) ')

    print('9    Convolution2D: 256,3,3  same relu ')
    print('9.5  Convolution2D: 256,3,3  same relu ')

    print('10   Merge[UpSampling2D(size=(2, 2) 9.5), 7.5] ')
    print('11   Convolution2D: 256,3,3  same relu ')
    print('11.5 Convolution2D: 256,3,3  same relu ')

    print('12   Merge[UpSampling2D(size=(2, 2) 9.5), 7.5] ')
    print('13   Convolution2D: 128,3,3  same relu ')
    print('13.5 Convolution2D: 128,3,3  same relu ')

    print('14   Merge[UpSampling2D(size=(2, 2) 9.5), 7.5] ')
    print('15   Convolution2D: 64,3,3  same relu ')
    print('15.5 Convolution2D: 64,3,3  same relu ')

    print('16   Merge[UpSampling2D(size=(2, 2) 9.5), 7.5] ')
    print('17   Convolution2D: 32,3,3  same relu ')
    print('17.5 Convolution2D: 32,3,3  same relu ')

    print('18   Convolution2D: 11,1,1  same relu ')
    print('Output: ')
    print('      layer: '+str(18)+ ' ')
    print('      tensor: (Batch_num,'+ str(PATCH_SIZE)+','+ str(PATCH_SIZE)+','+ str(N_ClASS) +') ')

    print('N_parames: 7,847,563')
    print("\n")

    print("Model details:")
    model.summary()
    print("Result details(Intersection over Union):")
    for idx in range(len(test_images_name)):
        print(test_images_name[idx]+': '+ '%.2f'%(inter_over_unions_list[idx]*100)+"% \t", end = '', flush = True)
    print("\n")
    
    sys.stdout = stdout
    log_file.close()  
    
if __name__ == '__main__':
    print('Hello')
else:
    print("function : 'log_write' can be used")