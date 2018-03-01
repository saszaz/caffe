import caffe
import numpy as np
from util import check_params, Map


class ArgmaxLayer(caffe.Layer):
    def setup(self, bottom, top):
      params = eval(self.param_str)
      check_params(params,
                   nn_num=3,
                   ntot=1024)
	        
      self.params = Map(params)
      self.batch_size = bottom[0].shape[0]
            
    def forward(self, bottom, top):
      assert len(bottom[0].shape)==2
      
      k=self.params.nn_num
      for itt in range(self.batch_size):        
        args = bottom[0].data[itt,:].argsort()[-k:][::-1]
        for nn_id in range(self.params.nn_num):
          top[nn_id].data[itt, :] = 0.
          top[nn_id].data[itt, args[nn_id]] = 1.
         

    def reshape(self, bottom, top):
      for nn_id in range(self.params.nn_num):
        top[nn_id].reshape(self.batch_size, self.params.ntot)

    def backward(self, top, propagate_down, bottom):
      pass
