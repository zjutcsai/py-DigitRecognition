# -*- coding: utf-8 -*-
import os
import h5py

class Saver:
    
    # save function
    def save(self, save_path, net):
        # get save path and model name
        model_dir, model_name = os.path.split(save_path)
        # create directory if not exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        f = h5py.File(save_path, 'w')
        for layer in net.layers:
            if layer.trainable:
                name_W = layer.name + ':W'
                name_b = layer.name + ':b'
                f.create_dataset(name_W, data=layer.W)
                f.create_dataset(name_b, data=layer.b)
        f.close()
    
    # restore function
    def restore(self, restore_path, net):
        # check model path
        if not os.path.exists(restore_path):
            raise Exception('path:{} is not exist.'.format(restore_path))
        # read dataset
        f = h5py.File(restore_path, 'r')
        for layer in net.layers:
            if layer.trainable:
                name_W = layer.name + ':W'
                name_b = layer.name + ':b'
                layer.W = f[name_W][:]
                layer.b = f[name_b][:]
        f.close()
        return net