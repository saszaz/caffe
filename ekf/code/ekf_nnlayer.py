import caffe
import numpy as np
from util import check_params, NN, DartDB, Map

class EKFNNLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        check_params(params, 
                compute_mask=False,
                use_traj_label=False,
                use_cam_pose=False,
                nn_root=None,
                nn_shape=None,
                nn_query_size=1,
                nn_num=1,
                nn_db_size=np.inf,
                nn_ignore=1)
	
	self.params = Map(params)
	self.nn_db = DartDB(self.params.nn_root, self.params.compute_mask, self.params.use_traj_label, self.params.use_cam_pose)
	self.nn = NN(self.nn_db, self.params.nn_db_size, self.params.nn_ignore, self.params.use_cam_pose)
 
	assert self.params.nn_num <= self.params.nn_query_size
	
    def reshape(self, bottom, top):
	#Reshape tops
	batch_size = bottom[0].shape[0]
	if self.params.use_cam_pose: 
	    assert (self.nn_db.jps.shape[1] + self.nn_db.cam_poses.shape[1]) == (bottom[0].shape[1] + bottom[1].shape[1])
	else:
	    assert self.nn_db.jps.shape[1] == bottom[0].shape[1]
	cur_top = 0
	for nn_id in range(self.params.nn_num):
		# print "nn_id = ", nn_id
	    top[cur_top + 0].reshape(batch_size, self.params.nn_shape[2], self.params.nn_shape[0], self.params.nn_shape[1])
	    top[cur_top + 1].reshape(batch_size, 1, self.params.nn_shape[0], self.params.nn_shape[1])
	    top[cur_top + 2].reshape(batch_size, self.nn_db.jps.shape[1])
	    top[cur_top + 3].reshape(batch_size, 1)
	    top[cur_top + 4].reshape(batch_size, 1)
	    top[cur_top + 5].reshape(batch_size, 1)
	    top[cur_top + 6].reshape(batch_size, 1)
	    cur_top += 7
#	    top[cur_top + 0].reshape(batch_size, 3, self.params.nn_shape[0], self.params.nn_shape[1])
#	    top[cur_top + 1].reshape(batch_size, 1, self.params.nn_shape[0], self.params.nn_shape[1])
#	    top[cur_top + 2].reshape(batch_size, self.nn_db.jps.shape[1])
#	    top[cur_top + 3].reshape(batch_size, 1)
#	    cur_top += 4
	    #self.top_names.extend(['nn_img_' + str(nn_id), 'nn_seg_' + str(nn_id)])
	    #self.top_names.append('nn_jp_' + str(nn_id))
	    #self.top_names.append('nn_w_' + str(nn_id))
     
    
    def forward(self, bottom, top):
        for itt in range(bottom[0].shape[0]):
	    jp = bottom[0].data[itt]
#	    nn_ids = self.nn.nn_ids(jp, self.params.nn_query_size)
     
	    if self.params.use_cam_pose: 
		cp = bottom[1].data[itt]
		jp = np.concatenate((jp,cp))
	    if self.params.use_traj_label: 
		nn_ids, nt_ids = self.nn.nt_ids(jp, self.params.nn_query_size)
	    else:
		nn_ids = self.nn.nn_ids(jp, self.params.nn_query_size)

#	    if hasattr(nn_ids, '__len__'):
#		nn_ids = np.random.choice(nn_ids, size=self.params.nn_num, replace=False)
#	    else:
#		nn_ids = [nn_ids]
     
	    assert len(nn_ids)==self.params.nn_query_size
     
#	    print ""
#	    print "nn_ids = ", nn_ids
#	    print "nt_ids = ", nt_ids
	    for i in range(len(nn_ids)):
		nn_id = nn_ids[i]
#		nn_id = nn_id[0]
		nn_jp, nn_img, nn_seg, nn_tl, nn_cp = self.nn_db.read_instance(nn_id, size=self.params.nn_shape,
                                                                 compute_mask=self.params.compute_mask,
                                                                 use_traj_label=self.params.use_traj_label,
                                                                 use_cam_pose=self.params.use_cam_pose)
#		print "neighbor: ", i
#		print "nn_jp: ", nn_jp
#		print "nn_tl: ", nn_tl
#		top[i * 4 + 0].data[itt, ...] = nn_img[0].transpose((2,0,1))
#		if nn_seg: top[i * 4 + 1].data[itt, ...] = nn_seg[0]
#		top[i * 4 + 2].data[itt, ...] = nn_jp
#		top[i * 4 + 3].data[itt, ...] = 1
#		top[i * 4 + 3].data[itt, ...] = nn_tl  
  
		top[i * 7 + 0].data[itt, ...] = nn_img[0].transpose((2,0,1))
		if nn_seg: 
          		top[i * 7 + 1].data[itt, ...] = nn_seg[0]
		else:
          		top[i * 7 + 1].data[itt, ...] = None
		top[i * 7 + 2].data[itt, ...] = nn_jp
		top[i * 7 + 3].data[itt, ...] = 1
		top[i * 7 + 4].data[itt, ...] = nn_tl
		top[i * 7 + 5].data[itt, ...] = nn_id
		top[i * 7 + 6].data[itt, ...] = nn_cp
  
  
#	    for j in range(len(nt_ids)):
#		top[i * 4 + 4 + j].data[itt, ...] = nt_ids[j]         
          

    def backward(self, top, propagate_down, bottom):
	bottom[0].diff[...] = 0

    
    def forward_jv(self, top, bottom):
	for top_id in range(self.params.nn_num * 4):
	    top[top_id].diff[...] = 0
	