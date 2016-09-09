import caffe

import numpy as np
import os
from matplotlib.pyplot import imshow, show, figure
from skimage import io
from skimage.transform import resize


def show(net, root_path = './result/', prefix = '', output=[('pimg', 'pred'), ('pseg', 'deconv9_segm'), ('img', 'img'), ('seg', 'seg'), ('nn_img', 'nn_img'), ('nn_seg', 'nn_seg')]):
    for name,top in output:
	if 'img' in name:
	    data = net.blobs[top].data.transpose((0, 2, 3, 1))
	elif 'seg' in name:
	    data = net.blobs[top].data
	    if data.shape[1] == 1:
		data = data[:, 0]
	    elif data.shape[1] == 2:
		edata = np.exp(data - data.max(1).reshape((data.shape[0], 1) + data.shape[2:]))
		data = edata[:,1]/edata.sum(1)
	
	batch_size = data.shape[0]
	for i in xrange(batch_size):
	    file_path = root_path + '%s_%d_%s.png' % (prefix, i, name)
	    data[i] *= 255
	    io.imsave(file_path, data[i].astype('uint8'))





model = '../test_flow.prototxt'
weights = '../snapshot/real_flow128_dof4_iter_40000.caffemodel'
# init
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(model, weights, caffe.TRAIN)


n_iter = 10000
delta = 1

for i in range(n_iter):
    print 'Iteration', i
    net.forward()
    if (i % delta) == 0:
	#show(net, prefix = str(i), output=[('img', 'img'), ('pimg','pred'), ('nn_img', 'nn_img')])
	show(net, prefix = str(i), output=[('img', 'img'), ('pimg','pred'), ('nn_img', 'nn_img')])
