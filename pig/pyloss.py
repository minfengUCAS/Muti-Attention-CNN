import caffe
import numpy as np

class DivLossLayer(caffe.Layer):
    """
    Compute div loss from ICCV2017 `Learning Multi-Attention Convolutional Neural 
    Network for Fine-Grained Image Recognition`
    """
    def setup(self, bottom, top):
        # check input
        if len(bottom) != 4:
            raise Exception("Need four inputs to compute distance")

    def reshape(self, bottom, top):
        # check input dimensions match        
        if bottom[0].data.shape[1] != 784:
            raise Exception("Bottom 1 must have the 784.")
        if bottom[1].data.shape[1] != 784:
            raise Exception("Bottom 2 must have the 784.")
        if bottom[2].data.shape[1] != 784:
            raise Exception("Bottom 3 must have the 784.")
        if bottom[3].data.shape[1] != 784:
            raise Exception("Bottom 4 must have the 784.")
     
        # difference is shape of inputs
        self.diff0 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff1 = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.diff2 = np.zeros_like(bottom[2].data, dtype=np.float32)
        self.diff3 = np.zeros_like(bottom[3].data, dtype=np.float32)
 
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        margin = 0.02

        self.diff0[...] = np.max(np.vstack([[bottom[1].data], [bottom[2].data], [bottom[3].data]]), axis=0) - margin
        self.diff1[...] = np.max(np.vstack([[bottom[0].data], [bottom[2].data], [bottom[3].data]]), axis=0) - margin
        self.diff2[...] = np.max(np.vstack([[bottom[0].data], [bottom[1].data], [bottom[3].data]]), axis=0) - margin
        self.diff3[...] = np.max(np.vstack([[bottom[0].data], [bottom[1].data], [bottom[2].data]]), axis=0) - margin

        top[0].data[...] = (np.sum(bottom[0].data*self.diff0) + np.sum(bottom[1].data*self.diff1) + 
                            np.sum(bottom[2].data*self.diff2) + np.sum(bottom[3].data*self.diff3)) / bottom[0].num


    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] =  self.diff0 / bottom[0].num
        bottom[1].diff[...] =  self.diff1 / bottom[1].num
        bottom[2].diff[...] =  self.diff2 / bottom[2].num
        bottom[3].diff[...] =  self.diff3 / bottom[3].num

class DisLossLayer(caffe.Layer):
    """
    Compute Dis loss from ICCV2017 `Learning Multi-Attention Convolutional Neural 
    Network for Fine-Grained Image Recognition`
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need four inputs to compute distance.")

    def get_maps(self, a, cx, cy, k=1.0):
        batch_size = cx.shape[0]
        maps = np.zeros([batch_size, a, a], dtype=np.float32)

        for b in range(batch_size):
            for i in range(a):
                for j in range(a):
                    maps[b][i][j] = k/(k + ((i-cx[b])*(i-cx[b]) + (j-cy[b])*(j-cy[b])))
        return maps.reshape(batch_size, a*a)

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].data.shape[1] != 28*28:
            raise Exception("Bottom 0 must have the 784.")
        if bottom[1].data.shape[1] != 28*28:
            raise Exception("Bottom 1 must have the 784.")
        if bottom[2].data.shape[1] != 28*28:
            raise Exception("Bottom 2 must have the 784.")
        if bottom[3].data.shape[1] != 28*28:
            raise Exception("Bottom 3 must have the 784.")
     
        # reshape bottom
        batch_size = bottom[0].data.shape[0]
        bottom[0].reshape(batch_size, 28*28)
        bottom[1].reshape(batch_size, 28*28)
        bottom[2].reshape(batch_size, 28*28)
        bottom[3].reshape(batch_size, 28*28)
        
        # difference is shape of inputs
        self.diff0 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff1 = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.diff2 = np.zeros_like(bottom[2].data, dtype=np.float32)
        self.diff3 = np.zeros_like(bottom[3].data, dtype=np.float32)

        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        num = np.argmax(bottom[0].data, axis=1)
        cx = num % 28
        cy = num / 28
        maps = self.get_maps(28, cx, cy)
        self.diff0[...] = bottom[0].data - maps

        num = np.argmax(bottom[1].data, axis=1)
        cx = num % 28
        cy = num / 28
        maps = self.get_maps(28, cx, cy)
        self.diff1[...] = bottom[1].data - maps
 
        num = np.argmax(bottom[2].data, axis=1)
        cx = num % 28
        cy = num / 28
        maps = self.get_maps(28, cx, cy)
        self.diff2[...] = bottom[2].data - maps

        num = np.argmax(bottom[3].data, axis=1)
        cx = num % 28
        cy = num / 28
        maps = self.get_maps(28, cx, cy)
        self.diff3[...] = bottom[3].data - maps

        top[0].data[...] = (np.sum(self.diff0*bottom[0].data) + np.sum(self.diff1*bottom[1].data) + 
                np.sum(self.diff2*bottom[1].data) + np.sum(self.diff3*bottom[3].data)) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] =  self.diff0 / bottom[0].num
        bottom[1].diff[...] =  self.diff1 / bottom[1].num
        bottom[2].diff[...] =  self.diff2 / bottom[2].num
        bottom[3].diff[...] =  self.diff3 / bottom[3].num
