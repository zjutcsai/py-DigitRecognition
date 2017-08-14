# -*- coding: utf-8 -*-
import cv2
import numpy as np

def next_batch(X, Y, offset, batch_size):
    batch_x = X[offset*batch_size:(offset+1)*batch_size] / 255.0
    batch_y = Y[offset*batch_size:(offset+1)*batch_size]
    return batch_x, batch_y

def _image_prepare(name, image_scale):
    im = cv2.imread(name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (image_scale,image_scale))
    return im
    
def next_batch_test(X, offset, batch_size, image_scale):
    batch_name = X[offset*batch_size:(offset+1)*batch_size]
    batch_x = np.array([_image_prepare(x, image_scale) for x in batch_name]) / 255.0
    return batch_x