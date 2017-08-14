# -*- coding: utf-8 -*-
import numpy as np
from config import cfg 
from tools.net import Net
from libs.dataset import download_mnist
from tools.sgdsolver import SGDSolver

if __name__ == '__main__':
    # set random seed
    np.random.RandomState(seed=cfg.seed)
    # get dataset
    dataset = download_mnist(cfg.data_path)
    # get network
    net = Net(**cfg) 
    # get solver, current support SGD only
    solver = SGDSolver(cfg)
    # train the network
    solver.train(net, dataset)
