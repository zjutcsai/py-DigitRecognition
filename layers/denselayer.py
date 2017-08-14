# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from .baselayer import BaseLayer

class DenseLayer(BaseLayer):

    def reshape(self):
        # fetch parameters
        self.name = self.args.get('name', 'fully_connected')
        self.input_nums = self.prev_layer.top_shape[-1]
        self.output_nums = self.args.get('output_nums', 128)
        # set trainable flag
        self.trainable = self.args.get('trainable', True)
        self.prev_grad_W = np.zeros((self.input_nums,self.output_nums),dtype=np.float32)
        self.prev_grad_b = np.zeros((self.output_nums),dtype=np.float32)
        # compute bottom shape && top shape
        self.bottom_shape = self.prev_layer.top_shape
        self.top_shape = [self.bottom_shape[0], self.output_nums]
        
        # print info
        print('[Construct] Layer: {}, input_shape: {}, output_shape: {}'.format(
              self.name, self.bottom_shape, self.top_shape))
    
    def initialize(self):
        # initialize weights and bias
        self.W = np.random.normal(0, 0.01, size=(self.input_nums,self.output_nums))
        self.b = np.zeros(self.output_nums, np.float32)
            
    def forward(self, bottom):
        # check input size
        assert list(bottom.shape) == list(self.bottom_shape)
        # forward compute
        self.bottom = bottom
        self.top = np.dot(self.bottom, self.W) + self.b
        return self.top
    
    def backward(self, grad_top):
        # fully connect layer(no activation function): 
        # y = x
        # x = pre_layer.top * w + b
        self.grad_top = grad_top
        # dloss/db = grad_top
        self.grad_b = np.sum(grad_top, axis=0)
        # dloss/dW = dloss/dx * dx/dW = grad_top * prev_layer.top
        # grad_top.shape: [n, c], prev_layer.top.shape: [n, o], W.shape: [o, c]
        # grad_W = prev_layer.top.T  *  grad_top  /n
        n, c = grad_top.shape
        self.grad_W = np.dot(self.bottom.transpose(), grad_top)
        # compute grad_bottom dloss/prev_layer.top
        # grad_bottom = grad_top * W.T
        self.grad_bottom = np.dot(self.grad_top, self.W.transpose())
        return self.grad_bottom