# -*- coding: utf-8 -*-
from easydict import EasyDict as edict

cfg = edict()
# base config
cfg.image_scale = 28
cfg.channels = 1
# train config
cfg.batch_size = 16
cfg.learning_rate = 0.01
cfg.momentum = 0.9
cfg.max_step = 10000
cfg.log_step = 100
cfg.test_step = 400
cfg.save_step = 2000
cfg.decay_step = 3000
cfg.weights = 0
cfg.seed = 0
cfg.save_path = 'data/model'
cfg.data_path = 'data/mnist'