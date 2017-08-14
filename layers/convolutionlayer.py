# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from .baselayer import BaseLayer
from libs.ops import op_conv2d, op_conv2d_backprop
from libs.ops import sliding_windows
from libs.ops import op_padding
from libs.ops import op_compute_top_shape
    
class ConvLayer(BaseLayer):
    
    def reshape(self):
        # fetch parameters
        self.name = self.args.get('name', 'conv_layer')
        self.input_nums = self.prev_layer.top_shape[-1]
        self.output_nums = self.args.get('output_nums', 64)
        self.kernel = self.args.get('kernel', 3)
        self.padding = self.args.get('padding', 'SAME')
        self.strides = self.args.get('strides', 1)
        # set trainable flag
        self.trainable = self.args.get('trainable', True)
        self.prev_grad_W = np.zeros((self.kernel,self.kernel,self.input_nums,self.output_nums),dtype=np.float32)
        self.prev_grad_b = np.zeros((self.output_nums),dtype=np.float32)
        # set bottom shape && compute top shape
        self.origin_bottom_shape = self.prev_layer.top_shape
        self.top_shape, self.padding_bottom_shape = op_compute_top_shape(self.prev_layer.top_shape,self.output_nums,\
                                                                         self.kernel,self.strides,self.padding)

        # print info
        print('[Construct] Layer: {}, input_shape: {}, output_shape: {}.'.format(
              self.name, self.origin_bottom_shape, self.top_shape, self.output_nums))
    
    def initialize(self):
        # initialize weights and bias
        self.W = np.random.normal(0, 0.01, size=(self.kernel,self.kernel,self.input_nums,self.output_nums))
        self.b = np.zeros(self.output_nums, np.float32)
    
    def forward(self, bottom):
        # check input size
        assert list(bottom.shape) == list(self.origin_bottom_shape)
        # padding or crop bottom feature maps
        self.bottom = op_padding(bottom, self.padding_bottom_shape)
        # convolutional operate
        self.top = op_conv2d(self.bottom, self.W, self.strides)
        self.top = self.top + self.b
        return self.top
    
    def backward(self, grad_top):
        # conv layer(no activation function): 
        # y = x
        # x = (sigma pre_layer.top * w) + b
        self.grad_top = grad_top
        
        # dloss/db = grad_top
        n, h, w, c = grad_top.shape
        self.grad_b = grad_top.reshape(n*h*w, c)
        self.grad_b = np.sum(self.grad_b, axis=0)

        # dloss/dW = sigma grad_top * prev_layer.top
        self.grad_W = np.zeros_like(self.W, dtype=np.float32)
        for patch, h_idx, w_idx in sliding_windows(grad_top, 1, 1, 1):
            patch = patch.reshape(n,c)
            patch_prev = self.bottom[:, h_idx*self.strides:h_idx*self.strides+self.kernel,\
                                     w_idx*self.strides:w_idx*self.strides+self.kernel, :]                      
            for n_idx in range(n):
                single_prev = patch_prev[n_idx]
                single_grad = patch[n_idx]
                single_grad_w = [single_prev*x for x in single_grad]
                single_grad_w = np.array(single_grad_w, dtype=np.float32).transpose(1,2,3,0)
                self.grad_W += single_grad_w
        #print(self.grad_W[:,:,0,0])        
        # compute grad_bottom && crop or fill for "padding" method
        self.grad_bottom = op_conv2d_backprop(grad_top, self.W, self.strides)
        if list(self.grad_bottom.shape) != list(self.origin_bottom_shape):
            self.grad_bottom = op_padding(self.grad_bottom, self.origin_bottom_shape)
        return self.grad_bottom