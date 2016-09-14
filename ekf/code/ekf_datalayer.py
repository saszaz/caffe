import caffe
import numpy as np
import random
from multiprocessing import Process, Queue, Lock
from util import check_params, NN, DartDB, shape_str, correct_hist, ImageFormatException
import scipy as sp
import sys, traceback

class DataLoaderProcess(Process):
    def __init__(self, name=None, args=(), kwargs=None):
        Process.__init__(self, name=name)
        self.queue = kwargs['queue']
        self.loader = kwargs['loader']

    def run(self):
	try:
	    while True:
		try:
		    item = self.loader.load_next_data()
		    self.queue.put(item)
		except ImageFormatException as e:
		    print 'I0913 21:37:16.779006 20177 ekf_datalayer.py] ' + str(e)
		    pass
	except Exception as e:
	    self.queue.put(None)
	    print str("".join(traceback.format_exception(*sys.exc_info())))
	    raise e
	    

class DataLoader(object):
    def __init__(self, params, db, nn_db):
	self.lock = Lock()
	self.db = db
	self.cur = db.length
        self.im_shape = params['im_shape']
        self.nn_shape = params['nn_shape']
        self.hist_eq = params['hist_eq']
        self.indexes = np.arange(db.length)
	self.shuffle = params['shuffle']
	self.subtract_mean = params['subtract_mean']
	if self.subtract_mean:
	    self.mean_img = self.db.read_mean_img(self.im_shape)
	
	self.im_shape = params['im_shape']
	self.load_nn = params['load_nn']
	self.nn_query_size = params['nn_query_size']
	if self.load_nn:
	    self.nn_db = nn_db
	    #nn_ignore = 1 if db.db_root == nn_db.db_root else 0
	    nn_ignore = 0
	    self.nn = NN(nn_db, params['nn_db_size'], nn_ignore)
   
    def load_next_data(self):
	nid = self.get_next_id()
        jp, imgs, segs = self.db.read_instance(nid, size=self.im_shape)
        item = {'jp':jp}
	for i in xrange(len(imgs)):
	    img = imgs[i]
	    if self.hist_eq:
		img = correct_hist(img)
	    item.update({'img_' + shape_str(self.im_shape[i]):img.transpose((2,0,1)), 'seg_' + shape_str(self.im_shape[i]): segs[i]})
	if self.load_nn:
	    nn_id = self.nn.nn_ids(jp, self.nn_query_size)
	    if hasattr(nn_id, '__len__'):
		nn_id = random.choice(nn_id)
	    nn_jp, nn_imgs, nn_segs = self.nn_db.read_instance(nn_id, size=self.nn_shape)
	    item.update({'nn_jp':nn_jp})
	    for i in xrange(len(nn_imgs)):
		nn_img = nn_imgs[i]
		if self.hist_eq:
		    nn_img = correct_hist(nn_img)
		item.update({'nn_img_' + shape_str(self.nn_shape[i]):nn_img.transpose((2,0,1)), 'nn_seg_' + shape_str(self.nn_shape[i]): nn_segs[i]})    
        return item

    def get_next_id(self):
	self.lock.acquire()
	if self.cur >= len(self.indexes) - 1:
            self.cur = 0
            if self.shuffle:
		random.shuffle(self.indexes)
	else:
	    self.cur += 1
	self.lock.release()
	return self.indexes[self.cur]

class EKFDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        check_params(params, 
                batch_size=1, 
                db_root=None, 
                nn_root='',
                subtract_mean=False,
                shuffle=True,
                num_threads = 1, 
                im_shape=[128, 128],
                nn_shape='',
                load_nn=False,
                nn_query_size=1,
                hist_eq=False,
                nn_db_size=np.inf)
	
	#im_shape and nn_shape should be list of image sizes [(128, 128), (256, 256)]
	if not hasattr(params['im_shape'][0], '__len__'):
	    params['im_shape'] = [params['im_shape']]
	    
	if params['nn_shape'] == '':
	    params['nn_shape'] = params['im_shape']
	
	if not hasattr(params['nn_shape'][0], '__len__'):
	    params['nn_shape'] = [params['nn_shape']]
	    
	self.load_nn = params['load_nn']
        self.batch_size = params['batch_size']
        self.num_threads = params['num_threads']
        self.queue = Queue(self.batch_size)
        self.processes = [None]*self.num_threads
	nn_db = None
        db = DartDB(params['db_root'])
        
	#Reshape tops
	cur_top = 0
	self.top_names = []
	for im_shape in params['im_shape']:
	    self.top_names.extend(['img_' + shape_str(im_shape), 'seg_' + shape_str(im_shape)])
	    top[cur_top].reshape(self.batch_size, 3, im_shape[0], im_shape[1])
	    top[cur_top + 1].reshape(self.batch_size, 1, im_shape[0], im_shape[1])
	    cur_top += 2
	self.top_names.append('jp')
        top[cur_top].reshape(self.batch_size, db.jps.shape[1])
        cur_top += 1
        if self.load_nn:
	    for nn_shape in params['nn_shape']:
		self.top_names.extend(['nn_img_' + shape_str(nn_shape), 'nn_seg_' + shape_str(nn_shape)])
		top[cur_top].reshape(self.batch_size, 3, nn_shape[0], nn_shape[1])
		top[cur_top + 1].reshape(self.batch_size, 1, nn_shape[0], nn_shape[1])
		cur_top += 2
	    self.top_names.append('nn_jp')
	    top[cur_top].reshape(self.batch_size, db.jps.shape[1])
	    cur_top += 1
	    if params['nn_root'] == '':
		params['nn_root'] = params['db_root']
		nn_db = db
	    else:
		nn_db = DartDB(params['nn_root'])
	#Initiate and start processes
        for i in range(self.num_threads):
            data_loader = DataLoader(params, db, nn_db)
            self.processes[i] = DataLoaderProcess(kwargs={'queue':self.queue,'loader':data_loader})
            self.processes[i].daemon = True
            self.processes[i].start()
	 
    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            item = self.queue.get()
            for idx in xrange(len(self.top_names)):
		top[idx].data[itt,...] = item[self.top_names[idx]]
	
    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass

    
