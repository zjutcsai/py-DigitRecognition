# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
from config import cfg 
from tools.net import Net
from libs.utils import next_batch
from libs.dataset import download_mnist
from tools.saver import Saver

def evaluate(net, test_x, test_y, batch_size):
    train_nums = test_x.shape[0]
    max_batches = train_nums // batch_size
    avg_accuracy = .0
    # compute batch accuracy, print
    for batch in range(max_batches):
        batch_x, batch_y = next_batch(test_x, test_y, batch, batch_size)
        # forward compute
        logits = net.step(batch_x)
        batch_accuracy = np.mean(np.array(np.argmax(logits,axis=1)==np.argmax(batch_y,axis=1), np.float32))
        # update global average acc
        avg_accuracy += batch_accuracy / max_batches
        print('Evaluation: {}/{}, batch accuracy: {}'.format(batch, max_batches, batch_accuracy))
    # print average accuray in test dataset
    print('Evaluation average accuracy: {}'.format(avg_accuracy))
    
if __name__ == '__main__':
    # get dataset
    _, _, test_images, test_labels = download_mnist(cfg.data_path)
    # get network
    net = Net(**cfg) 
    # get saver
    saver = Saver()
    # restore the network
    saver.restore(os.path.join(cfg.save_path,'net.h5'), net)
    # evaluate in test set
    evaluate(net, test_images, test_labels, cfg.batch_size)
