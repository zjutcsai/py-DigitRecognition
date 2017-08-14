# -*- coding: utf-8 -*-
import numpy as np

class Loss: 
    @staticmethod
    def SoftMax_CrossExtropy_Loss(logits, labels):
        #compute loss
        #loss = - sigma labels_i * log(logits_i)
        loss = -np.sum(labels * np.log(logits+1e-8), axis=1)
        reduce_loss = np.mean(loss)

        return reduce_loss
    
    @staticmethod
    def L2_Loss(diffs):
        batch = diffs.shape[0]
        diffs = diffs.reshape(batch, -1)
        loss = 0.5 * np.sum(diffs * diffs, axis=1)
        reduce_loss = np.mean(loss)
        
        return reduce_loss        