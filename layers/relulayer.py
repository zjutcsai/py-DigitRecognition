# -*- coding: utf-8 -*-
import numpy as np
from .baselayer import BaseLayer

class ReLULayer(BaseLayer):
    
    def reshape(self):
        # fetch parameters
        self.name = self.args.get('name', 'relu')
        # set trainable flag
        self.trainable = self.args.get('trainable', False)
        # set bottom shape && compute top shape
        self.bottom_shape = self.prev_layer.top_shape
        self.top_shape = self.bottom_shape
        # print info
        print('[Construct] Layer: {}, input_shape: {}, output_shape: {}.'.format(self.name, self.bottom_shape, self.top_shape))
    
    def initialize(self):
        pass
    
    def forward(self, bottom):
        self.bottom = bottom
        self.top = (self.bottom>=0).astype(np.float32) * self.bottom
        return self.top
    
    def backward(self, grad_top):
        # set top gradient
        self.grad_top = grad_top
        # compute bottom gradient
        self.grad_bottom = (self.bottom>=0).astype(np.float32) * grad_top
        
        return self.grad_bottom