import numpy as np
import caffe
from caffe import layers as L, params as P
import util

def create_conv(n, conv_filler, conv_param, num_output, ip_filler, ip_param, engine=P.Convolution.CAFFE):
    conv_args = dict()
    conv_args.update(conv_filler)
    conv_args.update(conv_param)

    ip_args = dict()
    ip_args.update(ip_filler)
    ip_args.update(ip_param)
    
    #Conv relu max 1
    #256x256
    n.conv1_1 = L.Convolution(n.img, kernel_size=3, pad=0,num_output=64, engine=engine,**conv_args)
    n.conv1_1_relu = L.ReLU(n.conv1_1, in_place=True)
    n.pool1 = L.Pooling(n.conv1_1_relu, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    
    #Conv relu max 2
    #128x128
    n.conv2_1 = L.Convolution(n.pool1, kernel_size=3, pad=0,num_output=128, engine=engine,**conv_args)
    n.conv2_1_relu = L.ReLU(n.conv2_1, in_place=True)
    n.pool2 = L.Pooling(n.conv2_1_relu, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    
    #Conv relu max 3
    #64x64
    n.conv3_1 = L.Convolution(n.pool2, kernel_size=3, pad=0,num_output=256, engine=engine,**conv_args)
    n.conv3_1_relu = L.ReLU(n.conv3_1, in_place=True)
    n.pool3 = L.Pooling(n.conv3_1_relu, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    
    #Conv relu max 4
    #32x32
    n.conv4_1 = L.Convolution(n.pool3, kernel_size=3, pad=0,num_output=512, engine=engine,**conv_args)
    n.conv4_1_relu = L.ReLU(n.conv4_1, in_place=True)
    n.pool4 = L.Pooling(n.conv4_1_relu, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    
    #Conv relu max 5
    #16x16
    n.conv5_1 = L.Convolution(n.pool4, kernel_size=3, pad=0,num_output=512, engine=engine,**conv_args)
    n.conv5_1_relu = L.ReLU(n.conv5_1, in_place=True)
    n.pool5 = L.Pooling(n.conv5_1_relu, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    
    #8x8
    n.fc6 = L.InnerProduct(n.pool5, num_output=1024, **ip_args)
    n.fc6_relu = L.ReLU(n.fc6, in_place=True)
    n.fc7 = L.InnerProduct(n.fc6_relu, num_output=1024, **ip_args)
    n.fc7_relu = L.ReLU(n.fc7, in_place=True)
    n.fc8 = L.InnerProduct(n.fc7_relu, num_output=num_output, **ip_args)    
    return n.fc8

def create_inner(n, jp, nn_jp, nn_img, ip_filler, ip_param):
    
    ip_args = dict()
    ip_args.update(ip_filler)
    ip_args.update(ip_param)
    
    if nn_jp is None:
	#InnerProducts
	n.fc1 = L.InnerProduct(jp, num_output=512, **ip_args)
	n.fc1_relu = L.ReLU(n.fc1, in_place=True)
	n.fc2 = L.InnerProduct(n.fc1_relu, num_output=512, **ip_args)
	n.fc2_relu = L.ReLU(n.fc2, in_place=True)
	n.fc3 = L.InnerProduct(n.fc2_relu, num_output=1024, **ip_args)
	n.fc3_relu = L.ReLU(n.fc3, in_place=True)
	n.fc4 = L.InnerProduct(n.fc3_relu, num_output=1024, **ip_args)
	n.fc4_relu = L.ReLU(n.fc4, in_place=True)
	n.fc5 = L.InnerProduct(n.fc4_relu, num_output=16384, **ip_args)
	n.fc5_relu = L.ReLU(n.fc5, in_place=True)
	output = n.fc5_relu
    elif nn_img is None:
	#merge jp and nn_jp
	n.fc1 = L.InnerProduct(jp, num_output=256, **ip_args)
	n.fc1_relu = L.ReLU(n.fc1, in_place=True)
	
	n.fc2 = L.InnerProduct(nn_jp, num_output=256, **ip_args)
	n.fc2_relu = L.ReLU(n.fc2, in_place=True)
	
	n.concat1 = L.Concat(n.fc1_relu, n.fc2_relu)
	
	n.fc3 = L.InnerProduct(n.concat1, num_output=512, **ip_args)
	n.fc3_relu = L.ReLU(n.fc3, in_place=True)
	
	n.fc4 = L.InnerProduct(n.fc3_relu, num_output=1024, **ip_args)
	n.fc4_relu = L.ReLU(n.fc4, in_place=True)
	n.fc5 = L.InnerProduct(n.fc4_relu, num_output=1024, **ip_args)
	n.fc5_relu = L.ReLU(n.fc5, in_place=True)
	n.fc6 = L.InnerProduct(n.fc5_relu, num_output=16384, **ip_args)
	n.fc6_relu = L.ReLU(n.fc6, in_place=True)
	output = n.fc6_relu
	
    else:
	#merge jp and nn_jp
	n.fc1 = L.InnerProduct(jp, num_output=256, **ip_args)
	n.fc1_relu = L.ReLU(n.fc1, in_place=True)
	
	n.fc2 = L.InnerProduct(nn_jp, num_output=256, **ip_args)
	n.fc2_relu = L.ReLU(n.fc2, in_place=True)
	n.concat1 = L.Concat(n.fc1_relu, n.fc2_relu)
	n.fc3 = L.InnerProduct(n.concat1, num_output=512, **ip_args)
	n.fc3_relu = L.ReLU(n.fc3, in_place=True)
	
	#merge nn_img
	n.fc4 = L.InnerProduct(nn_img, num_output=1024, **ip_args)
	n.fc4_relu = L.ReLU(n.fc4, in_place=True)
	n.fc5 = L.InnerProduct(n.fc4_relu, num_output=1024, **ip_args)
	n.fc5_relu = L.ReLU(n.fc5, in_place=True)
	n.concat2 = L.Concat(n.fc5_relu, n.fc3_relu)
	
	#nonlinearity to get the output
	n.fc6 = L.InnerProduct(n.concat2, num_output=1024, **ip_args)
	n.fc6_relu = L.ReLU(n.fc6, in_place=True)
	n.fc7 = L.InnerProduct(n.fc6_relu, num_output=1024, **ip_args)
	n.fc7_relu = L.ReLU(n.fc7, in_place=True)
	n.fc8 = L.InnerProduct(n.fc7_relu, num_output=16384, **ip_args)
	n.fc8_relu = L.ReLU(n.fc8, in_place=True)
	output = n.fc8_relu

    n.reshape = L.Reshape(output, reshape_param=dict(shape=dict(dim=[-1, 256, 8, 8])))
	
#input_top dim = [batch_size, 256, 8, 8]
def create_deconv(net_info, n, input_top, deconv_filler, conv_filler, deconv_param, conv_param, img_nout = 2,  engine=P.Convolution.CAFFE):
    conv_args = dict()
    conv_args.update(conv_filler)
    conv_args.update(conv_param)
    
    #Deconv Conv 6
    n.deconv6 = L.Deconvolution(input_top, convolution_param=dict(pad=1,kernel_size=4, stride=2, num_output=256, **deconv_filler), **deconv_param)
    n.deconv6_relu = L.ReLU(n.deconv6, in_place=True)
    
    n.conv6_1 = L.Convolution(n.deconv6_relu, kernel_size=3, pad=1,num_output=256, engine=engine,**conv_args)
    n.conv6_1_relu = L.ReLU(n.conv6_1, in_place=True)
    
    #Deconv Conv 7
    n.deconv7 = L.Deconvolution(n.conv6_1_relu, convolution_param=dict(pad=1,kernel_size=4, stride=2, num_output=128, **deconv_filler), **deconv_param)
    n.deconv7_relu = L.ReLU(n.deconv7, in_place=True)
    
    n.conv7_1 = L.Convolution(n.deconv7_relu, kernel_size=3, pad=1,num_output=128, engine=engine,**conv_args)
    n.conv7_1_relu = L.ReLU(n.conv7_1, in_place=True)
    
    #Deconv Conv 8
    n.deconv8 = L.Deconvolution(n.conv7_1_relu, convolution_param=dict(pad=1,kernel_size=4, stride=2, num_output=64, **deconv_filler), **deconv_param)
    n.deconv8_relu = L.ReLU(n.deconv8, in_place=True)
    
    n.conv8_1 = L.Convolution(n.deconv8_relu, kernel_size=3, pad=1,num_output=64, engine=engine,**conv_args)
    n.conv8_1_relu = L.ReLU(n.conv8_1, in_place=True)
    
    pimg = None
    psegm = None
    if np.all(net_info['datalayer_param']['im_shape'] == [64, 64]):
	#pimg, psegm
	n.pimg_64 = L.Convolution(n.conv8_1_relu, kernel_size=3, pad=1,num_output=img_nout, engine=engine,**conv_args)
	pimg = n.pimg_64
	if net_info['predict_seg']:
	    n.psegm_64 = L.Convolution(n.conv8_1_relu, kernel_size=3, pad=1,num_output=img_nout, engine=engine,**conv_args)
	    psegm = n.psegm_64
	return (pimg, psegm)
    
    #Deconv Conv 9
    n.deconv9 = L.Deconvolution(n.conv8_1_relu, convolution_param=dict(pad=1,kernel_size=4, stride=2, num_output=32, **deconv_filler), **deconv_param)
    n.deconv9_relu = L.ReLU(n.deconv9, in_place=True)
    
    n.conv9_1 = L.Convolution(n.deconv9_relu, kernel_size=3, pad=1,num_output=32, engine=engine,**conv_args)
    n.conv9_1_relu = L.ReLU(n.conv9_1, in_place=True)
    
    
    if np.all(net_info['datalayer_param']['im_shape'] == [128, 128]):
	#pimg, psegm
	n.pimg_128 = L.Convolution(n.conv9_1_relu, kernel_size=3, pad=1,num_output=img_nout, engine=engine,**conv_args)
	pimg = n.pimg_128
	if net_info['predict_seg']:
	    n.psegm_128 = L.Convolution(n.conv9_1_relu, kernel_size=3, pad=1,num_output=img_nout, engine=engine,**conv_args)
	    psegm = n.psegm_128
	return (pimg, psegm)
    
    #Deconv Conv 9
    n.deconv10 = L.Deconvolution(n.conv9_1_relu, convolution_param=dict(pad=1,kernel_size=4, stride=2, num_output=16, **deconv_filler), **deconv_param)
    n.deconv10_relu = L.ReLU(n.deconv10, in_place=True)
    
    n.conv10_1 = L.Convolution(n.deconv10_relu, kernel_size=3, pad=1,num_output=16, engine=engine,**conv_args)
    n.conv10_1_relu = L.ReLU(n.conv10_1, in_place=True)
	
    if np.all(net_info['datalayer_param']['im_shape'] == [256, 256]):	
	#pimg, psegm
	n.pimg_256 = L.Convolution(n.conv10_1_relu, kernel_size=3, pad=1,num_output=img_nout, engine=engine,**conv_args)
	pimg = n.pimg_256
	if net_info['predict_seg']:
	    n.psegm_256 = L.Convolution(n.conv10_1_relu, kernel_size=3, pad=1,num_output=img_nout, engine=engine,**conv_args)
	    psegm = n.psegm_256
	return (pimg, psegm)
    
    raise Exception('Unsupported Output Size!')
    

def create_coupled_net(net_info, n, input_top, deconv_filler, conv_filler, deconv_param, conv_param, engine=P.Convolution.CAFFE):
    conv_args = dict()
    conv_args.update(conv_filler)
    conv_args.update(conv_param)
    
    #Deconv Conv 6
    n.deconv6 = L.Deconvolution(input_top, convolution_param=dict(pad=1,kernel_size=4, stride=2, num_output=256, **deconv_filler), **deconv_param)
    n.deconv6_relu = L.ReLU(n.deconv6, in_place=True)
    
    n.conv6_1 = L.Convolution(n.deconv6_relu, kernel_size=3, pad=1,num_output=256, engine=engine,**conv_args)
    n.conv6_1_relu = L.ReLU(n.conv6_1, in_place=True)
    
    #Deconv Conv 7
    n.deconv7 = L.Deconvolution(n.conv6_1_relu, convolution_param=dict(pad=1,kernel_size=4, stride=2, num_output=128, **deconv_filler), **deconv_param)
    n.deconv7_relu = L.ReLU(n.deconv7, in_place=True)
    
    n.conv7_1 = L.Convolution(n.deconv7_relu, kernel_size=3, pad=1,num_output=128, engine=engine,**conv_args)
    n.conv7_1_relu = L.ReLU(n.conv7_1, in_place=True)
    
    #Deconv Conv 8
    n.deconv8 = L.Deconvolution(n.conv7_1_relu, convolution_param=dict(pad=1,kernel_size=4, stride=2, num_output=64, **deconv_filler), **deconv_param)
    n.deconv8_relu = L.ReLU(n.deconv8, in_place=True)
    
    n.conv8_1 = L.Convolution(n.deconv8_relu, kernel_size=3, pad=1,num_output=64, engine=engine,**conv_args)
    n.conv8_1_relu = L.ReLU(n.conv8_1, in_place=True)
    
    #warp img_64 and add it to the feature set
    n.disp_64 = L.Convolution(n.conv8_1_relu, kernel_size=3, pad=1,num_output=2, engine=engine,**conv_args)
    n.w64 = L.FlowWarping(n.nn_img_64, n.disp_64, flow_warping_param = dict(scale = 128/25.6))
    n.conv8_1_concat = L.Concat(n.conv8_1_relu, n.w64)

    #Deconv Conv 9
    n.deconv9 = L.Deconvolution(n.conv8_1_concat, convolution_param=dict(pad=1,kernel_size=4, stride=2, num_output=32, **deconv_filler), **deconv_param)
    n.deconv9_relu = L.ReLU(n.deconv9, in_place=True)
    
    n.conv9_1 = L.Convolution(n.deconv9_relu, kernel_size=3, pad=1,num_output=32, engine=engine,**conv_args)
    n.conv9_1_relu = L.ReLU(n.conv9_1, in_place=True)
    
    #warp img_128 and add it to the feature set
    n.disp_128 = L.Convolution(n.conv9_1_relu, kernel_size=3, pad=1,num_output=2, engine=engine,**conv_args)
    n.w128 = L.FlowWarping(n.nn_img_128, n.disp_128, flow_warping_param = dict(scale = 256/25.6))
    n.conv9_1_concat = L.Concat(n.conv9_1_relu, n.w128)
    
    n.pimg_128 = L.Convolution(n.conv9_1_concat, kernel_size=3, pad=1,num_output=3, engine=engine,**conv_args)
    pimg = n.pimg_128
    return pimg

def create_net(net_info):
    deconv_filler = dict(weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0.0))
    conv_filler = dict(weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0.0))
    ip_filler = dict(weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0.1))
    
    deconv_param = dict(param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=2, decay_mult=0)])
    conv_param = dict(param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=2, decay_mult=0)])
    ip_param = dict(param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=2, decay_mult=0)])
    
    n = caffe.NetSpec()
    if net_info['name'] == 'chair':
	#Datalayer
	n.img, n.seg, n.jp = L.Python(module='ekf_datalayer', layer='EKFDataLayer', ntop=3, param_str=str(net_info['datalayer_param']))
	if not net_info['predict_seg']:
	    n.silence_seg = L.Silence(n.seg, ntop=0)
	create_inner(n, n.jp, None, None, ip_filler, ip_param)
	pimg, psegm = create_deconv(net_info, n, n.reshape, deconv_filler, conv_filler, deconv_param, conv_param, 3)
	#Loss
	n.img_loss = L.EuclideanLoss(pimg, n.img, loss_weight = 10.0)
	
	if net_info['predict_seg']:
	    n.msk_loss = L.SoftmaxWithLoss(psegm, n.seg, loss_weight = 1000.0)
    elif net_info['name'] == 'flow':
	#Datalayer
	n.img, n.seg, n.jp, n.nn_img, n.nn_seg, n.nn_jp = L.Python(module='ekf_datalayer', layer='EKFDataLayer', ntop=6, param_str=str(net_info['datalayer_param']))
	n.silence_seg = L.Silence(n.seg, ntop=0)
	n.silence_nn_seg = L.Silence(n.nn_seg, ntop=0)
	
	#Network
	#create_conv(n, conv_filler, conv_param)
	create_inner(n, n.jp, n.nn_jp, None, ip_filler, ip_param)
	pimg, psegm = create_deconv(net_info, n, n.reshape, deconv_filler, conv_filler, deconv_param, conv_param)
	n.pred = L.FlowWarping(n.nn_img, pimg, flow_warping_param = dict(scale = sum(net_info['datalayer_param']['im_shape'])/25.6))
	
	#Loss
	n.img_loss = L.EuclideanLoss(n.pred, n.img, loss_weight = 1.0)
	
	if net_info['predict_seg']:
	    n.msk_loss = L.SoftmaxWithLoss(psegm, n.seg, loss_weight = 10.0)
    elif net_info['name'] == 'coupled':
	n.img, n.seg, n.jp, n.nn_img_64, n.nn_seg_64, n.nn_img_128, n.nn_seg_128, n.nn_jp = L.Python(module='ekf_datalayer', layer='EKFDataLayer', ntop=8, param_str=str(net_info['datalayer_param']))
	n.silence_seg = L.Silence(n.seg, ntop=0)
	n.silence_nn_seg_64 = L.Silence(n.nn_seg_64, ntop=0)
	n.silence_nn_seg_128 = L.Silence(n.nn_seg_128, ntop=0)

	create_inner(n, n.jp, n.nn_jp, None, ip_filler, ip_param)
	pimg = create_coupled_net(net_info, n, n.reshape, deconv_filler, conv_filler, deconv_param, conv_param)
	n.img_loss = L.EuclideanLoss(pimg, n.img, loss_weight = 1.0)
    return n


def main():
    
    #train_datalayer_param = dict(load_nn=True, batch_size=256, db_root='/home/amirreza/caffe/ekf/data/data_ntot_129600', num_threads=4, im_shape=[128, 128], nn_query_size=100, shuffle=True)
    #test_datalayer_param = dict(load_nn=True, batch_size=1, db_root='/home/amirreza/caffe/ekf/data/data_ntot_360', nn_root='/home/amirreza/caffe/ekf/data/data_ntot_129600', num_threads=1, i
				
    #train_datalayer_param = dict(load_nn=True, batch_size=128, db_root='/nethome/ashaban6/caffe/ekf/data/data_ntot_823543', nn_root='/nethome/ashaban6/caffe/ekf/data/data_ntot_823543_sample_150k', num_threads=6, im_shape=[256, 256], nn_query_size=20, shuffle=True)
    #test_datalayer_param = dict(load_nn=True, batch_size=1, db_root='/nethome/ashaban6/caffe/ekf/data/data_ntot_823543', nn_root='/nethome/ashaban6/caffe/ekf/data/data_ntot_823543_sample_150k', num_threads=1, im_shape=[256, 256], nn_query_size=1, shuffle=False)
    name = 'flow'
    
    train_datalayer_param = dict(load_nn=True, batch_size=128, db_root='/nethome/ashaban6/caffe/ekf/data/last_version/reald/db_train_1', nn_root='/nethome/ashaban6/caffe/ekf/data/last_version/reald/db_train_1', num_threads=6, im_shape=[256, 256], nn_shape=[256, 256], nn_query_size=10, shuffle=True, hist_eq=False)
    test_datalayer_param = dict(load_nn=False, batch_size=1, db_root='/nethome/ashaban6/caffe/ekf/data/db_test', nn_root='/nethome/ashaban6/caffe/ekf/data/db_train', num_threads=1, im_shape=[256, 256], nn_shape=[256, 256], nn_query_size=1, shuffle=False,hist_eq=False)
    
    #train_datalayer_param = dict(load_nn=False, batch_size=128, db_root='/nethome/ashaban6/caffe/ekf/data/last_version/data0_ntot_100k_random', num_threads=2, im_shape=[128, 128], shuffle=True, hist_eq=False)
    #test_datalayer_param = dict(load_nn=False, batch_size=1, db_root='/nethome/ashaban6/caffe/ekf/data/db_test', nn_root='/nethome/ashaban6/caffe/ekf/data/db_train', num_threads=1, im_shape=[256, 256], nn_shape=[256, 256], nn_query_size=1, shuffle=False,hist_eq=False)
    #Create Train Network
    net_info = dict(name=name, predict_seg=False, datalayer_param=train_datalayer_param)
    net = create_net(net_info)
    util.save_net('../train_' + str(train_datalayer_param['im_shape'][0]) + '_' + net_info['name'] + '.prototxt', str(net.to_proto()))
    #Create Test Network
    net_info = dict(name=name, predict_seg=False, datalayer_param=test_datalayer_param)
    net = create_net(net_info)
    util.save_net('../test_' + str(test_datalayer_param['im_shape'][0]) + '_' + net_info['name'] + '.prototxt', str(net.to_proto()))

if __name__ == '__main__':
    main()
    
    
    
    
