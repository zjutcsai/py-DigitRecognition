# -*- coding: utf-8 -*-
import numpy as np
from layers import InputLayer
from layers import ConvLayer
from layers import ReLULayer
from layers import MaxPoolLayer
from layers import FlattenLayer
from layers import DenseLayer
from layers import SoftMaxLayer

class Net:
    def __init__(self, **args):
        # solve for mnist classification
        self.x_dim = args.get('image_scale', 28)
        self.batch_size = args.get('batch_size', 16)
        self.c_dim = args.get('channels', 1) 
        self.build_model()
    
    def build_model(self):
        layers = []
        input_shape = np.array([self.batch_size,self.x_dim,self.x_dim,self.c_dim])
        # layer_1: input_layer ==> [n, 28, 28, 1]
        x = InputLayer(input_shape)
        layers.append(x)
        # layer_2: conv_layer [n, 28, 28, 1] ==> [n, 28, 28, 32]
        x = ConvLayer(x, output_nums=20, kernel=5, strides=1, padding='SAME', name='conv1')
        layers.append(x)
        # layer_4: avgpool_layer [n, 28, 28, 32] ==> [n, 14, 14, 32]
        x = MaxPoolLayer(x, kernel=2, strides=2, paddind='SAME', name='pool1')
        layers.append(x)
        # layer_5: conv_layer [n, 14, 14, 32] ==> [n, 14, 14, 64]
        x = ConvLayer(x, output_nums=50, kernel=5, strides=1, padding='SAME', name='conv2')
        layers.append(x)
        # layer_7: avgpool_layer [n, 14, 14, 64] ==> [n, 7, 7, 64]
        x = MaxPoolLayer(x, kernel=2, strides=2, padding='SAME', name='pool2')
        layers.append(x)
        # layer_8: flatten_layer [n, 7, 7, 64] ==> [n, 7*7*64]
        x = FlattenLayer(x, name='flatten')
        layers.append(x)
        # layer_9: fullconnected_layer [n, 3136] ==> [n, 500]
        x = DenseLayer(x, output_nums=500, name='dense1')
        layers.append(x)
        # layer_10: relu_layer [n, 500] ==> [n, 500]
        x = ReLULayer(x, name='relu1')
        layers.append(x)
        # layer_11: fullconnected_layer [n, 500] ==> [n, 10]
        x = DenseLayer(x, output_nums=10, name='dense2')
        layers.append(x)
        # layer_12: softmax_layer [n, 10] ==> [n, 10]
        x = SoftMaxLayer(x, name='softmax')
        layers.append(x)
        
        self.layers = layers
    
    def step(self, x):
        for layer in self.layers[1:]:
            x = layer.forward(x)
        return x
    
    def step_inverse(self, g):
        for layer in self.layers[::-1]:
            g = layer.backward(g)
        
        