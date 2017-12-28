import caffe
import numpy as np
import cv2
import csv
import decimal
import math
import os

caffe.set_device(0)
caffe.set_mode_gpu()

ctx = decimal.Context()
ctx.prec = 8

def float_to_str(f):
    dl = ctx.create_decimal(repr(f))
    return format(dl, 'f')

#translate an img to fit the input of a network
def data_trans(img, shape):
	mu = np.array([109.973,127.338,123.883])
	transformer = caffe.io.Transformer({'data': shape})
	transformer.set_transpose('data', (2,0,1))  
	transformer.set_mean('data', mu)			
	transformer.set_raw_scale('data', 255)	 
	# transformer.set_channel_swap('data', (2,1,0)) 
	transformed_image = transformer.preprocess('data', img)
	return transformed_image
 
#crop the centor part from 448*n (keep image ratio) to 448*448
def crop_centor(img):
	[n,m,_]=img.shape
	if m>n:
		m = m*448/n
		n = 448
	else:
		n = n*448/m
		m=448
	return data_trans(cv2.resize(img,(m,n))/255.0,(1,3,n,m))[:,(n-448)/2:(n+448)/2,(m-448)/2:(m+448)/2]

# mu = array([109.973,127.338,123.883])
model_weights ='./model/fix_channel_group_1210.caffemodel'
model_def ='./deploy/pig_part_deploy_test.prototxt'
net = caffe.Net(model_def,model_weights,caffe.TEST)

test_list = open('./data/test_list.txt').readlines()
res_dict = {}
for test_file in test_list:
    print test_file
    img = cv2.imread(test_file.split(' ')[0])
    if img.ndim<3:
	img = np.transpose(np.array([img,img,img]),(1,2,0))
    [n,m,_]=img.shape
    data = crop_centor(img)
    net.blobs['data'].data[...] = data
    out = net.forward()
    p1 = net.blobs['prob1'].data
    p2 = net.blobs['prob2'].data
    p3 = net.blobs['prob3'].data
    p4 = net.blobs['prob4'].data
    p = np.vstack([p1,p2,p3,p4])
    p = np.mean(p, axis=0)
    filename = os.path.basename(test_file.split(' ')[0])
    res_dict[filename] = p

with open('result.csv', 'wb') as write_f:
    w = csv.writer(write_f)
    for name, val in res_dict.items():
        for i, p in enumerate(val):
            w.writerow([name, i+1, float_to_str(p/1.00000001)])


