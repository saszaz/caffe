##### call net.forwardJ and computes "computed_j"
##### compute true jacobian using backward method and computes "true_j" 
##### check whether computed_j[:, :, ::inc] == true_j[:, :, ::inc]

import sys
import time

## 1) remove old caffe from python path
#if '/home/amir/caffe/python' in sys.path:
#    sys.path.remove('/home/amir/caffe/python')
#
## 2) add new caffe to the python path
#sys.path.append('/home/amir/codes/clones/caffe/python/')

# 3) add $NEW_CAFFE/ekf folder to the python path
# It is only required if you want to compair ForwardJv with BackwardJv
sys.path.append('/home/sasha/kaffe/ekf')


import caffe
import numpy as np


# 4) set parameters
######################################
model = './deploy.prototxt'
weights = './barrett_iter_25986.caffemodel'

#From input (layer, bottom)
start = 'fc1'
bottom = 'label'

#To output (layer, top)
end = 'eltwise'
top = 'segm_img'

seed = 1234323
inc = 1000
######################################


net = caffe.Net(model, weights, caffe.TEST)
caffe.set_mode_gpu()

np.random.seed(seed)
net.blobs['label'].data[...] = np.random.uniform(low=-1.0, high=1.0, size=net.blobs['label'].data.shape)



t0 = time.time()
net.forward()
forward_time = time.time() - t0


t0 = time.time()
computed_j = net.forwardJ(start,end,bottom,top)
forwardj_time = time.time() - t0

assert net.blobs[top].shape[0] == 1
n = net.blobs[top].shape[0]
n_in = reduce(lambda x, y: x*y, net.blobs[bottom].shape)
n_out = reduce(lambda x, y: x*y, net.blobs[top].shape)


true_j = np.zeros((n, n_out, n_in))

t0 = time.time()
for i in range(0, n_out, inc):
    net.backward()
    #assert net.blobs[top].diff.flat[i] == 1 and net.blobs[top].diff.sum() == 1
    true_j[0, i][...] = net.blobs[bottom].diff.flat
backwardj_time = time.time() - t0
avg_err = inc * np.abs((computed_j - true_j)[0, ::inc]).sum()/n_in/n_out
print 'Forward time:', forward_time, 'Forwardj_time:', forwardj_time, 'Aprox. Backwardj_time:', backwardj_time * inc
print 'AVG Error:', avg_err




