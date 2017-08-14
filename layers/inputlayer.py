# -*- coding: utf-8 -*-
from .baselayer import BaseLayer

class InputLayer(BaseLayer):
    def __init__(self, input_shape):
        self.top_shape = input_shape
        self.reshape()
        
    # forward function
    def forward(self, bottom):
        pass
    
    # backward function
    def backward(self, grad_top):
        pass
    
    def reshape(self):
        self.trainable = False
        print('[Construct] Layer: input, output_shape: {}'.format(self.top_shape))
    
    def initialize(self):
        pass
    