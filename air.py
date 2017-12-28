import caffe
import numpy as np
import cv2

def part_box(net):
	part1 = np.argmax(net.blobs['mask_1_4'].data.reshape(28,28))
	part2 = np.argmax(net.blobs['mask_2_4'].data.reshape(28,28))
	part3 = np.argmax(net.blobs['mask_3_4'].data.reshape(28,28))
	part4 = np.argmax(net.blobs['mask_4_4'].data.reshape(28,28))
	return part1 % 28, part1 / 28, part2 % 28, part2 / 28, part3 % 28, part3 / 28, part4 % 28, part4 / 28

def get_part(img,parts):
	img_parts = [[] for i in range(4)]
	[n,m,_]=img.shape
	if m>n:
		islong=1
	else:
		islong=0
	for i in range(4):
		if islong==0:
			parts[i][0] =  parts[i][0]*m/28+8
			parts[i][1] =  parts[i][1]*m/28+8 + (n - m) / 2
			l =  48*m/448
		else:
			parts[i][0] =  parts[i][0]*n/28+8 + (m - n) / 2
			parts[i][1] =  parts[i][1]*n/28+8
			l =  48*n/448
		box = (np.maximum(0,np.int(parts[i][0] - l)), np.maximum(0,np.int(parts[i][1] - l)),
		 np.minimum(m,np.int(parts[i][0] + l)), np.minimum(n,np.int(parts[i][1] + l)))
		img_parts[i] = cv2.resize(img[box[1]:box[3],box[0]:box[2],:],(224,224))
	return img_parts


def data_trans(img, shape):
	mu = np.array([109.973,127.338,123.883])
	transformer = caffe.io.Transformer({'data': shape})
	transformer.set_transpose('data', (2,0,1))  
	transformer.set_mean('data', mu)			
	transformer.set_raw_scale('data', 255)	 
	# transformer.set_channel_swap('data', (2,1,0)) 
	transformed_image = transformer.preprocess('data', img)
	return transformed_image


def to_square(img):
	[n,m,_]=img.shape
	if m>n:
		new_img = np.zeros((m,m,3)) + 127
		new_img[(m-n)/2:(m+n)/2,:,:] = img
	else:
		new_img = np.zeros((n,n,3)) + 127
		new_img[:,(n-m)/2:(m+n)/2,:] = img
	return new_img

def crop_lit_centor(img):
	[n,m,_]=img.shape
	if m>n:
		m = m*256/n
		n = 256
	else:
		n = n*256/m
		m=256
	return data_trans(cv2.resize(img,(m,n))/255.0,(1,3,n,m))[:,(n-224)/2:(n+224)/2,(m-224)/2:(m+224)/2]




ls = 3333
caffe.set_mode_gpu()
caffe.set_device(14)
# mu = array([109.973,127.338,123.883])
model_weights ='model/air_part.caffemodel'
model_def ='deploy/air_part_deploy.prototxt'
p_net = caffe.Net(model_def,model_weights,caffe.TEST)

model_weights ='model/air_class_iter_0.caffemodel'
model_def ='deploy/air_cls_deploy.prototxt'
c_net = caffe.Net(model_def,model_weights,caffe.TEST)

test_list = open('air_data/test_list.txt').readlines()
accuracy = 0
# feature = np.zeros((ls,2560))
for i in range(ls):	
	print i
	if i>-1:
		img = cv2.imread('air_data/' + test_list[i].split(' ')[0])
		if img.ndim<3:
			img = np.transpose(np.array([img,img,img]),(1,2,0))
		label = np.int(test_list[i].split(' ')[1])
		data = to_square(img)
		p_net.blobs['data'].data[...] = data_trans(cv2.resize(data,(448,448))/255.0,(1,3,448,448))
		p_out = p_net.forward()
		part_boxs = part_box(p_net)
		img_part = get_part(data,np.array(part_boxs).reshape((4,2)))

		c_net.blobs['label'].data[...] = label
		c_net.blobs['ori_data'].data[...] = crop_lit_centor(data)
		c_net.blobs['part1_data'].data[...] = data_trans(img_part[0]/255.0,(1,3,224,224))
		c_net.blobs['part2_data'].data[...] = data_trans(img_part[1]/255.0,(1,3,224,224))
		c_net.blobs['part3_data'].data[...] = data_trans(img_part[2]/255.0,(1,3,224,224))
		c_net.blobs['part4_data'].data[...] = data_trans(img_part[3]/255.0,(1,3,224,224))
		
		c_out = c_net.forward()

		# feature[i] = c_net.blobs['pin'].data.reshape(2560)

		accuracy = accuracy + c_net.blobs['accuracy'].data
		# print (c_net.blobs['accuracy'].data)
# np.savetxt('\\\\msralab\\ProjectData\\MSMDATA-3\\jianf\\share\\for.heliang\\code_for_release\\iccv_2017\\data\\air_f_train1.txt',feature)

print (accuracy/3333)


