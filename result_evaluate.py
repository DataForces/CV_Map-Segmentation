#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 22:00:19 2017

@author: kaku
"""


def intersection_over_union(y_true, y_pred, smooth):
    """
    Calculate intersection over union value
    similar as jaccard_coef
    Returns intersection over union value ranges from 0~1
    
    Parameters
    ----------
        y_true : np.array 
            ground truth
        y_pred : np.array
            predittd result
    
    Returns
    -------
        inter_over_union: float 
            ranges from 0~1
    """
    pred_flat, true_flat = y_pred.flatten(), y_true.flatten()
    intersection = list(pred_flat == true_flat).count(True)
    union = len(pred_flat) + len(true_flat)
    inter_over_union = round((intersection + smooth) / (union - intersection + smooth), 4)
    return inter_over_union

def overall_accuracy(y_true, y_pred):
    """
    Calculate average precision score
    """
    pred_flat, true_flat = y_pred.flatten(), y_true.flatten()
    intersection = list(pred_flat == true_flat).count(True)
    sum_ = len(true_flat)
    accuracy = round(intersection/sum_, 4)
    return accuracy

if __name__ == '__main__':
    print('Hello')
else:
    print("Evaluation functions can be used")
    
    
    