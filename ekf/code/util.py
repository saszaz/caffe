import numpy as np
import h5py
import os.path as osp
import os
from skimage import img_as_float, exposure
from skimage.io import imread, imsave
from skimage.transform import resize
import random
from scipy.spatial import KDTree
import os
import errno
from shutil import copyfile

debug_mode = False
def cprint(string, style = None):
    if not debug_mode:
	return
    if style is None:
	print str(string)
    else:
	print style + str(string) + bcolors.ENDC

class ImageFormatException(Exception):
    def __init__(self, e):
	self.e = e

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#defaults is a list of (key, val) is val is None key is required field
def check_params(params, **kwargs):
    for key, val in kwargs.items():
	key_defined = (key in params.keys())
	if val is None:
	    assert key_defined, 'Params must include {}'.format(key)
	elif not key_defined:
	    params[key] = val

def correct_intensity(img):
    p2 = np.percentile(img, 2)
    p98 = np.percentile(img, 98)
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale

def correct_hist(img):
    img_heq = np.zeros(img.shape)
    img_heq[:,:,0]=exposure.equalize_hist(img[:,:,0])
    img_heq[:,:,1]=exposure.equalize_hist(img[:,:,1])
    img_heq[:,:,2]=exposure.equalize_hist(img[:,:,2])
    return img_heq

def save_net(net_path, proto):
    with open(net_path, 'w') as f:
        f.write(proto)

def sample_db(src, dst, max_sample=np.inf, size=None):
    db = DartDB(src)
    db.sample(max_sample).save(dst,size)

def shape_str(shape):
    return str(shape[0]) + 'x' + str(shape[1])

def resize_img(img, size=None):
    if size is not None and np.any(np.array(size) != img.shape[:2]):
	return resize(img, size)
    return img
class DartDB:
    def __init__(self, db_root=None):
	self.db_root = db_root
	if db_root is not None:
	    self.init_db()
	
    #reads data and initialize self.jps, self.img_paths, self.length
    def init_db(self):
	self.jps = np.genfromtxt(osp.join(self.db_root, 'joints.txt'), delimiter=',')
	try:
	    self.dyn_data = np.genfromtxt(osp.join(self.db_root, 'dyn_data.txt'), delimiter=',')
	except:
	    pass
     	try:
	    self.traj_labels = np.genfromtxt(osp.join(self.db_root, 'traj_labels.txt'), delimiter=',')
	except:
	    pass
	self.img_paths = [f for f in os.listdir(self.db_root) if f.startswith('fr_') and f.endswith('.png')]
	img_nums = np.array([int(f[3:-4]) for f in self.img_paths])
	self.length = self.jps.shape[0]
	if self.traj_labels: assert len(self.length) == self.traj_labels.shape[0]
	num_order = np.argsort(img_nums)
	assert len(img_nums) == self.length and np.all(img_nums[num_order] == np.arange(0,self.length))
	self.img_paths = [self.img_paths[i] for i in num_order]

    def read_instance(self, indx, size=None, compute_mask=False, use_traj_label=False):
	jp = self.jps[indx]
	img_path = self.img_paths[indx]
	img = imread(osp.join(self.db_root, img_path))
	img = img_as_float(img)
	if not (img.max() <= 1.0 and img.min() >= 0 and len(img.shape) == 3 and img.shape[2] == 3):
	    print img.max(), img.min(), img.shape
	    e = Exception('Image has a wrong format: ' + osp.join(self.db_root, img_path))
	    raise ImageFormatException(e)
	    
	if size is None or not hasattr(size[0], '__len__'): 
	    images = (resize_img(img, size),)
	else:
	    images = ()
	    for s in size:
		images += (resize_img(img, s),)
	if compute_mask and not use_traj_label:
	    masks = ()
	    for img in images:
		mask = np.zeros(img.shape[:-1])
		mask[img.sum(2) > 2.99990] = 1
		masks += (mask,)
	    return (jp, images, masks)
	elif use_traj_label and not compute_mask:
	    assert self.traj_labels
	    traj_label = self.traj_labels[indx]
	    return (jp, images, None, traj_label)
	else:
	    return (jp, images)
	
    def read_mean_img(self, size=None):
	img = imread(osp.join(self.db_root, 'mean.png'))
	img = img_as_float(img)
	img = resize_img(img, size)
	if not (img.max() <= 1.0 and img.min() >= 0 and len(img.shape) == 3 and img.shape[2] == 3):
	    raise Exception('Image has a wrong format: ' + osp.join(self.db_root, img_path)) 
	return img
    
    def sample(self, max_sample=np.inf):
	data_ids = np.array(random.sample(np.arange(self.length), min(max_sample, self.length)), dtype='int')
	return self.__sample(data_ids)
    
    def split(self, set_size=None):
	assert sum(set_size) == self.length
	piv = np.concatenate(([0], np.cumsum(set_size)))
	dbs = []
	for i in xrange(len(set_size)):
	    sample_ids = range(piv[i], piv[i+1])
	    dbs.append(self.__sample(sample_ids))
	return dbs

    def __sample(self, sample_ids):
	db = DartDB()
	db.db_root = self.db_root
	db.jps = self.jps[sample_ids]
	if hasattr(self, 'dym_data'):
	    db.dym_data = self.dym_data
	db.img_paths = [self.img_paths[i] for i in sample_ids]
	db.length = len(sample_ids)
	return db
    
    def save(self, save_root, size = None):
	os.makedirs(save_root)
	try:
	    mean_img = self.read_mean_img(size=size)
	    imsave(osp.join(save_root, 'mean.png'), mean_img)
	except:
	    pass
	
	for i in range(self.length):
	    dst = osp.join(save_root, 'fr_%07d.png' % i)
	    if size is None:
		src = osp.join(self.db_root, self.img_paths[i])
		copyfile(src, dst)
	    else:
		jp, img = self.read_instance(i, size, False);
		imsave(dst, img[0])
	np.savetxt(osp.join(save_root, 'joints.txt'), self.jps, fmt='%1.10f', delimiter=', ')
    
    def extend(self, db):
	old_ind = int(db.length/2)
	jp_old, img_old = db.read_instance(old_ind, size=None, compute_mask=False)
	
	self.jps = np.concatenate((self.jps, db.jps))
	np.savetxt(osp.join(self.db_root, 'joints.txt'), self.jps, fmt='%1.10f', delimiter=', ')
	if hasattr(self, 'dym_data'):
	    self.dym_data = np.concatenate((self.dym_data, db.dym_data))
	    np.savetxt(osp.join(self.db_root, 'dyn_data.txt'), self.dym_data, fmt='%1.10f', delimiter=', ')
	for src_id in range(db.length):
	    dst_id = src_id + self.length
	    self.img_paths.append('fr_%07d.png' % dst_id)
	    src = osp.join(db.db_root, db.img_paths[src_id])
	    dst = osp.join(self.db_root, self.img_paths[dst_id])
	    copyfile(src, dst)
	
	
	jp_new, img_new = self.read_instance(self.length + old_ind, size=None, compute_mask=False)
	assert np.all(jp_new == jp_old) and np.all(img_old[0] == img_new[0])
	    
	self.length = len(self.img_paths)
	
class NN:
    def __init__(self, db, max_sample=np.inf, nn_ignore=1):
	self.db = db
	self.nn_ignore = nn_ignore
	self.max_sample = max_sample
	self.create_db()
	
    def create_db(self):
	self.data_ids = np.array(random.sample(np.arange(self.db.length), min(self.max_sample, self.db.length)), dtype='int')
	self.kdtree = KDTree(data=self.db.jps[self.data_ids].copy())
	
    def nn_ids(self, jp, nn = 1):
	d, i = self.kdtree.query(jp, k=nn+self.nn_ignore, eps=0, p=2)
	if hasattr(i, '__len__'):
	    return self.data_ids[i[self.nn_ignore:].squeeze()]
	else:
	    assert (nn + self.nn_ignore) == 1
	    return self.data_ids[i]
    def nt_ids(self, jp, nt=2, nn_start=10, nn_delta=10, max_iter=100): # Nearest-trajectory 
      assert nt>=2
      traj_found=[]
    
      nt_ids=self.nn_ids(jp,1)
      nt_ids=[nt_ids]
      nn_jp, nn_img, nn_seg, nn_tl = self.db.read_instance(nn_ids[0],size=self.params.nn_shape,compute_mask=False,use_traj_label=True)
      
      nn_num=nn_start
      nn_iter=0
	for i in range(2,nt):
	  j=0
       while j < max_iter: 
	   nt_ids=self.nn_ids(jp,nn_num)
        nn_num+=nn_delta
         
        
        
   
        
class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]
