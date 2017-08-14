# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from .baselayer import BaseLayer

class SoftMaxLayer(BaseLayer):
    
    def reshape(self):
        # set trainable flag
        self.name = self.args.get('name', 'softmax')
        self.trainable = self.args.get('trainable', False)
        # set bottom shape && compute top shape
        self.bottom_shape = self.prev_layer.top_shape
        self.top_shape = self.bottom_shape
        # print info
        print('[Construct] Layer: {}, input_shape: {}, output_shape: {}.'.format(self.name, self.bottom_shape, self.top_shape))
     
    def initialize(self):
        pass
    
    def forward(self, bottom):
        # check input size
        assert list(bottom.shape) == list(self.bottom_shape)
        # minus max value in order to exp(x) overflow
        n, c = self.bottom_shape
        max_value = np.max(bottom, axis=1, keepdims=True)
        max_value = np.repeat(max_value, c, axis=1)
        fixed_x = bottom - max_value
        # compute exp(x_i)
        exp_x = np.exp(fixed_x)
        exp_sum = np.sum(exp_x, axis=1, keepdims=True)
        exp_sum = np.repeat(exp_sum, c, axis=1)
        self.bottom = fixed_x
        self.top = exp_x / exp_sum
        return self.top
    
    def backward(self, labels):
        # compute bottom gradient
        # softmax gradient: 
        # dloss/dx_i = h_i - y_i
        self.grad_bottom = self.top - labels
        return self.grad_bottom