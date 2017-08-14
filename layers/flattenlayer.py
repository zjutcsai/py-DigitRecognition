# -*- coding: utf-8 -*-
import numpy as np
from .baselayer import BaseLayer

class FlattenLayer(BaseLayer):
    
    def reshape(self):
        # fetch parameters
        self.name = self.args.get('name', 'flatten')
        # set trainable flag
        self.trainable = self.args.get('trainable', False)
        # set bottom shape && compute top shape
        self.bottom_shape = self.prev_layer.top_shape
        self.top_shape = [self.bottom_shape[0], np.prod(self.bottom_shape[1:])]
        # print info
        print('[Construct] Layer: {}, input_shape: {}, output_shape: {}.'.format(self.name, self.bottom_shape, self.top_shape))
    
    def initialize(self):
        pass
    
    def forward(self, bottom):
        self.bottom = bottom
        self.top = bottom.reshape(self.top_shape)
        return self.top
    
    def backward(self, grad_top):
        # set top gradient
        self.grad_top = grad_top
        # set bottom gradient, equals to top grdient
        bottom_shape = self.bottom.shape
        self.grad_bottom = np.reshape(grad_top, bottom_shape)
        
        return self.grad_bottom