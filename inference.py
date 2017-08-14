# -*- coding: utf-8 -*-
from __future__ import division
import os
import argparse
import numpy as np
from glob import glob
from config import cfg 
from tools.net import Net
from libs.utils import next_batch_test
from tools.saver import Saver

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', dest='test_dir', type=str, help='Test images directory.')
    args = parser.parse_args()
    return args
    
def test(net, img_dir, batch_size, image_scale):
    # get image list
    image_list = glob(os.path.join(img_dir,'*'))
    
    max_batches = len(image_list)/batch_size
    # compute batch accuracy, print
    results = []
    for batch in range(max_batches):
        batch_x = next_batch_test(image_list, batch, batch_size, image_scale)
        # forward compute
        logits = net.step(batch_x)
        predict = np.argmax(logits,axis=1)
        results += predict.to_list()
    # write result into file
    with open(os.path.join(img_dir,'result.txt'), 'w') as f:
        for name, result in zip(image_list, results):
            print('Image name: {}, predict label: {}'.format(name, result))
            f.write(name + '  '+ result+'\n')
    print('Predict result has saved in {}'.format(os.path.join(img_dir,'result.txt')))
    
if __name__ == '__main__':
    # get test image dirctory
    args = parse_arg()
    # get network
    net = Net(**cfg) 
    # get saver
    saver = Saver()
    # restore the network
    saver.restore(os.path.join(cfg.save_path,'net.h5'), net)
    # evaluate in test set
    test(net, args.test_dir, cfg.batch_size, cfg.image_scale)
