# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
from .loss import Loss
from .saver import Saver
from libs.utils import next_batch

class SGDSolver:
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.weights = args.weights
        self.momentum = args.momentum
        self.max_step = args.max_step
        self.log_step = args.log_step
        self.test_step = args.test_step
        self.save_step = args.save_step
        self.decay_step = args.decay_step
        self.save_path = args.save_path
    
    def initialize_net_variables(self, net):
        for layer in net.layers:
            layer.initialize()
    
    def evaluate(self, net, test_x, test_y):
        max_test_groups = 10
        accuracy = 0.
        for group in range(max_test_groups):
            # fetch data
            batch_x, batch_y = next_batch(test_x, test_y, group, net.batch_size)
            # forward compute
            logits = net.step(batch_x)
            acc = np.mean(np.array(np.argmax(logits,axis=1)==np.argmax(batch_y,axis=1), np.float32))
            accuracy += acc / max_test_groups
        print('[*Evaluate] {} batches, average accuracy:{}'.format(max_test_groups, accuracy))
        
    
    def train(self, net, dataset):
        # get saver
        saver = Saver()
        
        # prepare network's params
        if self.weights == 0:
            self.initialize_net_variables(net)
        else:
            net = saver.restore(os.path.join(self.save_path,'net.h5'), net)
        
        # fetch train and test data
        train_x, train_y, test_x, test_y = dataset
        offset = 0
        max_batch = train_x.shape[0] // net.batch_size
        # train step
        for iters in range(self.max_step):
            # get batch of training data
            batch_x, batch_y = next_batch(train_x, train_y, offset, net.batch_size)
            # compute gradient
            logits = self.compute_gradient(net, batch_x, batch_y)
            # apply gradient
            self.apply_gradient(net)
            
            if iters % self.log_step == 0:
                # compute loss
                accuracy = np.mean(np.array(np.argmax(logits, axis=1)==np.argmax(batch_y,axis=1), np.float32))               
                batch_loss = Loss.SoftMax_CrossExtropy_Loss(logits, batch_y)
                print('[Train] Iterations: {}, loss: {}, train_acc: {}'.format(iters, batch_loss, accuracy))
            if iters % self.test_step == 0:
                self.evaluate(net, test_x, test_y)
            if iters % self.save_step == 0:
                saver.save(os.path.join(self.save_path,'net.h5'), net)
            if iters != 0 and iters % self.decay_step == 0:
                self.learning_rate /= 2
                
            # update offset
            offset = (offset + 1) % max_batch
    
    # compute gradient of current batch
    def compute_gradient(self, net, batch_x, batch_y):
        # forward compute
        logits = net.step(batch_x)
        # backward update
        net.step_inverse(batch_y)
        return logits
    
    # apply gradient of the net
    def apply_gradient(self, net):
        for layer in net.layers:
            if layer.trainable:
                # update weights
                delta_W = self.momentum*layer.prev_grad_W - self.learning_rate*layer.grad_W / net.batch_size
                layer.W = layer.W + delta_W
                layer.prev_grad_W = delta_W
                
                # update bias
                delta_b = self.momentum*layer.prev_grad_b - self.learning_rate*layer.grad_b / net.batch_size
                layer.b = layer.b + delta_b
                layer.prev_grad_b = delta_b
                
        
