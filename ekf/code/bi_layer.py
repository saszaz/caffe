import json
import time
import caffe
import numpy as np


class BILayer(caffe.Layer):
    def setup(self, bottom, top):
	self.v = 0
	params = eval(self.param_str)
	self.inc = params['inc']
	top[0].reshape(bottom[0].data.shape[0], 1)
	pass
    def forward(self, bottom, top):
	pass
    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        shape = bottom[0].diff.shape
        assert shape[0] == 1
        n_out = reduce(lambda x, y: x*y, shape)
        diff = np.zeros((n_out,))
        diff[self.v] = 1
        self.v += self.inc
        bottom[0].diff[...] = diff.reshape(bottom[0].diff.shape)
        
        
