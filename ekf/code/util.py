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
	self.img_paths = [f for f in os.listdir(self.db_root) if f.startswith('fr_') and f.endswith('.png')]
	img_nums = np.array([int(f[3:-4]) for f in self.img_paths])
	self.length = self.jps.shape[0]
	num_order = np.argsort(img_nums)
	assert np.all(img_nums[num_order] == np.arange(0,self.length))
	self.img_paths = [self.img_paths[i] for i in num_order]

    def read_instance(self, indx, size=None, compute_mask=True):
	jp = self.jps[indx]
	img_path = self.img_paths[indx]
	img = imread(osp.join(self.db_root, img_path))
	img = img_as_float(img)
	if not (img.max() <= 1.0 and img.min() >= 0 and len(img.shape) == 3 and img.shape[2] == 3):
	    raise Exception('Image has a wrong format: ' + osp.join(self.db_root, img_path))
	if size is None or not hasattr(size[0], '__len__'): 
	    images = (resize_img(img, size),)
	else:
	    images = ()
	    for s in size:
		images += (resize_img(img, s),)
	if compute_mask:
	    masks = ()
	    for img in images:
		mask = np.zeros(img.shape[:-1])
		mask[img.sum(2) > 2.99990] = 1
		masks += (mask,)
	    return (jp, images, masks)
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
	db = DartDB()
	db.db_root = self.db_root
	data_ids = np.array(random.sample(np.arange(self.length), min(max_sample, self.length)), dtype='int')
	db.jps = self.jps[data_ids]
	db.img_paths = [self.img_paths[i] for i in data_ids]
	db.length = len(data_ids)
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
	np.savetxt(osp.join(save_root, 'joints.txt'), self.jps, fmt='%1.6f', delimiter=', ')
    
class NN:
    def __init__(self, db, max_sample=np.inf):
	self.db = db
	self.max_sample = max_sample
	self.create_db()
	
    def create_db(self):
	self.data_ids = np.array(random.sample(np.arange(self.db.length), min(self.max_sample, self.db.length)), dtype='int')
	self.kdtree = KDTree(data=self.db.jps[self.data_ids].copy())
	
    def nn_ids(self, jp, nn = 1):
	d, i = self.kdtree.query(jp, k=nn+1, eps=0, p=2)
	return self.data_ids[i[1:].squeeze()]

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