# -*- coding: utf-8 -*-

class BaseLayer:
    def __init__(self, prev_layer, **args):
        self.args = args
        self.prev_layer = prev_layer
        self.reshape()
    
    # initilize ops
    def reshape(self):
        return NotImplementedError
    
    def initialize(self):
        return NotImplementedError
        
    # forward function
    def forward(self, bottom):
        return NotImplementedError
    
    # backward function
    def backward(self, grad_top):
        return  NotImplementedError
    
    
        
