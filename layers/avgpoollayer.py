# -*- coding: utf-8 -*-
from __future__ import division
from .baselayer import BaseLayer
from libs.ops import op_conv2d_separate
from libs.ops import op_conv2d_separate_backprop 
from libs.ops import op_padding 
from libs.ops import op_compute_top_shape

class AvgPoolLayer(BaseLayer):
    
    def reshape(self):
        # fetch parameters
        self.name = self.args.get('name', 'avg_pool')
        self.output_nums = self.prev_layer.top_shape[-1]
        self.kernel = self.args.get('kernel', 2)
        self.padding = self.args.get('padding', 'SAME')
        self.strides = self.args.get('strides', 2)
        # set trainable flag
        self.trainable = self.args.get('trainable', False)
        # set bottom shape && compute top shape
        self.origin_bottom_shape = self.prev_layer.top_shape
        self.top_shape, self.padding_bottom_shape = op_compute_top_shape(self.prev_layer.top_shape,self.output_nums,\
                                                                         self.kernel,self.strides,self.padding)
        print('[Construct] Layer: {}, input_shape: {}, output_shape: {}.'.format(
              self.name, self.origin_bottom_shape, self.top_shape))
    
    def initialize(self):
        pass
        
    def forward(self, bottom):
        # check input size
        assert list(bottom.shape) == list(self.origin_bottom_shape)
        # padding or crop feature maps
        self.bottom = op_padding(bottom, self.padding_bottom_shape)
        # convolutional operate
        self.top = op_conv2d_separate(self.bottom, self.kernel, self.strides, 'avgpool')
        return self.top
        
    def backward(self, grad_top):
        # set top gradient
        self.grad_top = grad_top
        # compute bottom gradient
        self.grad_bottom = op_conv2d_separate_backprop(grad_top, self.bottom, self.kernel,self.strides,'avgpool')
        # gradient feature maps crop or fill for "padding" method
        if self.grad_bottom != self.origin_bottom_shape:
            self.grad_bottom = op_padding(self.grad_bottom, self.origin_bottom_shape)
            
        return self.grad_bottom