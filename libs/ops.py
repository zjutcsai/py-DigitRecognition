# -*- coding: utf-8 -*-
import numpy as np
try:
    import reduce
except:
    from functools import reduce

# generate iteratable sliding windows
def sliding_windows(x, kh, kw, strides):
    n, h, w, ci = x.shape
    for h_idx in range(0, h-kh+1, strides):
        for w_idx in range(0, w-kw+1, strides):
            patch = x[:, h_idx:h_idx+kh, w_idx:w_idx+kw, :]
            yield patch, h_idx, w_idx
    
# standard convolve
def op_conv2d(x, weights, strides):
    n, h, w, ci = x.shape
    kh, kw, ci, co = weights.shape
    oh = (h - kh) // strides + 1
    ow = (w - kw )// strides + 1
    outs = np.zeros((n,oh,ow,co), dtype=np.float32)
    for patch, h_idx, w_idx in sliding_windows(x, kh, kw, strides):
        for out_map in range(co):
            conv = patch * weights[...,out_map]
            conv = conv.reshape(n,kh*kw*ci)
            conv = np.sum(conv, axis=1)
            outs[:, h_idx//strides, w_idx//strides, out_map] = conv
    return outs

# standard convolve-transpose, used for backpropgation
def op_conv2d_backprop(x, weights, strides):
    n, h, w, co = x.shape
    kh, kw, ci, co = weights.shape
    oh = (h - 1) * strides + kh
    ow = (w - 1) * strides + kw
    outs = np.zeros((n,oh,ow,ci))
    for patch, h_idx, w_idx in sliding_windows(x, 1, 1, 1):
        for batch in range(n):
            deconv = weights * patch[batch,...]
            # add all maps's gradient
            deconv = deconv.transpose(3,0,1,2)
            deconv = reduce(lambda x,y: x+y, deconv)
            outs[batch,h_idx*strides:h_idx*strides+kh,w_idx*strides:w_idx*strides+kw,:] += deconv
    return outs

# channel separate convolve, use for average & max pooling
def op_conv2d_separate(x, kernel, strides, conv_type='avgpool'):
    n, h, w, ci = x.shape
    oh = (h - kernel)//strides + 1
    ow = (w - kernel)//strides + 1
    outs = np.zeros((n,oh,ow,ci), dtype=np.float32)
    # convolve and set weights
    for patch, h_idx, w_idx in sliding_windows(x, kernel, kernel, strides):
        patch = patch.reshape(n,kernel*kernel,ci)
        if conv_type == 'avgpool':
            conv = np.mean(patch, axis=1)
        else:
            conv = np.max(patch, axis=1)
        outs[:,h_idx//strides,w_idx//strides,:] = conv
    return outs
    
# channel separate convolve-transpose, used for average & max pooling backpropgation
def op_conv2d_separate_backprop(x, x_prev, kernel, strides, conv_type='avgpool'):
    n, h, w, co = x.shape
    oh = (h - 1) * strides + kernel
    ow = (w - 1) * strides + kernel
    outs = np.zeros((n,oh,ow,co), dtype=np.float32)
    for patch, h_idx, w_idx in sliding_windows(x, 1, 1, 1):
        # compute deconv
        patch = np.squeeze(patch, axis=1)
        deconv = np.concatenate([patch]*kernel**2, axis=1).reshape(n,kernel,kernel,co)
        # compute mask
        if conv_type == 'avgpool':
            mask = np.ones((n,kernel,kernel,co), dtype=np.float32) * (1./kernel**2)
        else:
            # get prev layer output, used for maxpool
            patch_prev = x_prev[:,h_idx*strides:h_idx*strides+kernel,w_idx*strides:w_idx*strides+kernel,:]
            max_values = patch_prev.reshape(n,kernel*kernel,co)
            # fetch max value elem's position
            max_values = np.max(max_values, axis=1, keepdims=True)
            max_values = np.concatenate([max_values]*kernel**2, axis=1).reshape(n,kernel,kernel,co)
            # generate pooling mask, also called kernel
            mask = (patch_prev>=max_values).astype(np.float32)
        # multiple mask for gradient backpropagation
        deconv = deconv * mask
        outs[:,h_idx*strides:h_idx*strides+kernel,w_idx*strides:w_idx*strides+kernel,:] += deconv
    return outs

# compute layer's top shape according to bottom shape, kernel,strides and padding method
def op_compute_top_shape(bottom_shape, output_nums,  kernel, strides, padding):
    n, b_h, b_w, b_c = bottom_shape
    # compute output maps shape
    if padding == 'VALID':
        t_h = int(np.floor((b_h-kernel)/strides+1))
        t_w = int(np.floor((b_w-kernel)/strides+1))
    else:
        t_h = int(np.ceil(b_h/strides))
        t_w = int(np.ceil(b_w/strides))
    # compute padding/crop bottom shape
    new_h = (t_h - 1) * strides + kernel
    new_w = (t_w - 1) * strides + kernel
    return [n,t_h,t_w,output_nums], [n,new_h,new_w,b_c]

# padding or crop feature maps for forward && backward procedure 
def op_padding(x, target_shape):
    n, o_h, o_w, c = x.shape
    n, p_h, p_w, c = target_shape
    padding_x = np.zeros(target_shape, np.float32)
    if p_h > o_h: 
        offset_l = (p_w - o_w) // 2
        offset_t = (p_h - o_h) // 2
        padding_x[:,offset_t:offset_t+o_h,offset_l:offset_l+o_w,:] = x
    else:       # VALID convoluve
        # crop
        offset_l = (o_w - p_w) // 2
        offset_t = (o_h - p_h) // 2
        padding_x = x[:,offset_t:p_h+offset_t, offset_l:offset_l+p_w, :]
    return padding_x