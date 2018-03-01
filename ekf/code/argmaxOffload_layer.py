import caffe
import numpy as np
from util import check_params, Map


## DOESN'T WORK -- NN arrays not stored in binary proto
# Need to read from NN database instead.
 
class ArgmaxOffloadLayer(caffe.Layer):
    def setup(self, bottom, top):
      params = eval(self.param_str)
      check_params(params,
                   nn_num=3,
                   img_size=196608,
                   ntot=1024)
	        
      self.params = Map(params)
      self.nn_imgs = np.zeros((self.params.img_size,self.params.ntot))
      self.nn_jps = np.zeros((7,self.params.ntot))
      self.batch_size = bottom[0].shape[0]
#      self.initialized = False
            
    def forward(self, bottom, top):
      assert len(bottom[0].shape)==2
#      assert self.initialized
      
      k=self.params.nn_num
      for itt in range(self.batch_size):        
        args = bottom[0].data[itt,:].argsort()[-k:][::-1]
        cur_top = 0
        for nn_id in range(self.params.nn_num):
          top[cur_top + 0].data[itt, :] = self.nn_imgs[:,args[nn_id]] 
          top[cur_top + 1].data[itt, :] = self.nn_jps[:,args[nn_id]]   
          cur_top += 2           

    def reshape(self, bottom, top):
      cur_top = 0
      for nn_id in range(self.params.nn_num):
        top[cur_top + 0].reshape(self.batch_size, self.params.img_size)
        top[cur_top + 1].reshape(self.batch_size, 7)   
        cur_top += 2

    def backward(self, top, propagate_down, bottom):
      pass
	