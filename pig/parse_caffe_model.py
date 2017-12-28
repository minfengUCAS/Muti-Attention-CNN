# coding:utf-8
# author:ChrisZZ
# description: 从caffemodel文件解析出网络训练信息，以类似train.prototxt的形式输出到屏幕
import caffe.proto.caffe_pb2 as caffe_pb2
#caffemodel_filename = '/home/chris/work/fuckubuntu/caffe-fast-rcnn/examples/mnist/lenet_iter_10000.caffemodel'
caffemodel_filename = '/home/minfeng.zhan/workspace/release/model/bird_part.caffemodel'
model = caffe_pb2.NetParameter()
f=open(caffemodel_filename, 'rb')
model.ParseFromString(f.read())
f.close()
layers = model.layer
print 'name: ' + model.name
layer_id=-1
for layer in layers:
    layer_id = layer_id + 1
    res=list()
    # name
    res.append('layer {')
    res.append('  name: "%s"' % layer.name)
    # type
    res.append('  type: "%s"' % layer.type)
    # bottom
    for bottom in layer.bottom:
        res.append('  bottom: "%s"' % bottom)
    # top
    for top in layer.top:
        res.append('  top: "%s"' % top)
    # loss_weight
    for loss_weight in layer.loss_weight:
        res.append('  loss_weight: ' + str(loss_weight))
    # param
    for param in layer.param:
        param_res = list()
        if param.lr_mult is not None:
            param_res.append('    lr_mult: %s' % param.lr_mult)
        if param.decay_mult!=1:
            param_res.append('    decay_mult: %s' % param.decay_mult)
        if len(param_res)>0:
            res.append('  param{')
            res.extend(param_res)
            res.append('  }')
    # lrn_param
    if layer.lrn_param is not None:
        lrn_res = list()
        if layer.lrn_param.local_size!=5:
            lrn_res.append('    local_size: %d' % layer.lrn_param.local_size)
        if layer.lrn_param.alpha!=1:
            lrn_res.append('    alpha: %f' % layer.lrn_param.alpha)
        if layer.lrn_param.beta!=0.75:
            lrn_res.append('    beta: %f' % layer.lrn_param.beta)
        NormRegionMapper={'0': 'ACROSS_CHANNELS', '1': 'WITHIN_CHANNEL'}
        if layer.lrn_param.norm_region!=0:
            lrn_res.append('    norm_region: %s' % NormRegionMapper[str(layer.lrn_param.norm_region)])
        EngineMapper={'0': 'DEFAULT', '1':'CAFFE', '2':'CUDNN'}
        if layer.lrn_param.engine!=0:
            lrn_res.append('    engine: %s' % EngineMapper[str(layer.lrn_param.engine)])
        if len(lrn_res)>0:
            res.append('  lrn_param{')
            res.extend(lrn_res)
            res.append('  }')
    #power_param
    if layer.power_param is not None:
        power_res = list()
        if float(layer.power_param.power) != 1.0:
            power_res.append('    power: %f' % layer.power_param.power)
        if float(layer.power_param.scale) != 1.0:
            power_res.append('    scale: %f' % layer.power_param.scale)
        if float(layer.power_param.shift) != 0.0:
            power_res.append('    shift: %f' % layer.power_param.shift)
        if len(power_res) > 0:
            res.append('  power_param{')
            res.extend(power_res)
            res.append('  }')

    # eltwise_param
    if layer.eltwise_param is not None:
        operation_mapper=['PROD','SUM','MAX']
        eltwise_res = list()
        if int(layer.eltwise_param.operation) != 1:
            eltwise_res.append('    operation: %s' % operation_mapper[int(layer.eltwise_param.operation)])
        if len(eltwise_res)>0:
            res.append('  eltwise_param{')
            res.extend(eltwise_res)
            res.append('  }')
    
    # transpose_param
    if layer.transpose_param is not None:
        dims_res = list()
        for dim in layer.transpose_param.dim:
            dims_res.append('    dim:%d'%dim)
        if len(dims_res)>0:
            res.append('  transpose_param{')
            res.extend(dims_res)
            res.append('  }')

    # reshape_param
    if layer.reshape_param is not None:
        if layer.reshape_param.shape is not None:
            dim_res = list()
            for dim in layer.reshape_param.shape.dim:
                dim_res.append('      dim:%d'%dim)
            if len(dim_res):
                res.append('  reshape_param{')
                res.append('    shape{')
                res.extend(dim_res)
                res.append('    }')
                res.append('  }')


    # include
    if len(layer.include)>0:
        include_res = list()
        includes = layer.include
        phase_mapper={
            '0': 'TRAIN',
            '1': 'TEST'
        }
        for include in includes:
            if include.phase is not None:
                include_res.append('    phase: %s' % phase_mapper[str(include.phase)])
        if len(include_res)>0:
            res.append('  include {')
            res.extend(include_res)
            res.append('  }')
    # transform_param
    if layer.transform_param is not None:
        transform_param_res = list()
        if layer.transform_param.scale!=1:
            transform_param_res.append('    scale: %s'%layer.transform_param.scale)
        if layer.transform_param.mirror!=False:
            transform_param.res.append('    mirror: ' + layer.transform_param.mirror)
        if len(transform_param_res)>0:
            res.append('  transform_param {')
            res.extend(transform_param_res)
            res.res.append('  }')
    # data_param
    if layer.data_param is not None and (layer.data_param.source!="" or layer.data_param.batch_size!=0 or layer.data_param.backend!=0):
        data_param_res = list()
        if layer.data_param.source is not None:
            data_param_res.append('    source: "%s"'%layer.data_param.source)
        if layer.data_param.batch_size is not None:
            data_param_res.append('    batch_size: %d'%layer.data_param.batch_size)
        if layer.data_param.backend is not None:
            data_param_res.append('    backend: %s'%layer.data_param.backend)
        if len(data_param_res)>0:
            res.append('  data_param: {')
            res.extend(data_param_res)
            res.append('  }')
    # convolution_param
    if layer.convolution_param is not None:
        convolution_param_res = list()
        conv_param = layer.convolution_param
        if conv_param.num_output!=0:
            convolution_param_res.append('    num_output: %d'%conv_param.num_output)
        if len(conv_param.kernel_size) > 0:
            for kernel_size in conv_param.kernel_size:
                convolution_param_res.append('    kernel_size: %d' % kernel_size)
        if len(conv_param.pad) > 0:
            for pad in conv_param.pad:
                convolution_param_res.append('    pad: %d' % pad)
        if len(conv_param.stride) > 0:
            for stride in conv_param.stride:
                convolution_param_res.append('    stride: %d' % stride)
        if conv_param.weight_filler is not None and conv_param.weight_filler.type!='constant':
            convolution_param_res.append('    weight_filler {')
            convolution_param_res.append('      type: "%s"'%conv_param.weight_filler.type)
            convolution_param_res.append('    }')
        if conv_param.bias_filler is not None and conv_param.bias_filler.type!='constant':
            convolution_param_res.append('    bias_filler {')
            convolution_param_res.append('      type: "%s"'%conv_param.bias_filler.type)
            convolution_param_res.append('    }')
        if len(convolution_param_res)>0:
            res.append('  convolution_param {')
            res.extend(convolution_param_res)
            res.append('  }')
    # pooling_param
    if layer.pooling_param is not None:
        pooling_param_res = list()
        if layer.pooling_param.kernel_size>0:
            pooling_param_res.append('    kernel_size: %d' % layer.pooling_param.kernel_size)
            pooling_param_res.append('    stride: %d' % layer.pooling_param.stride)
            pooling_param_res.append('    pad: %d' % layer.pooling_param.pad)
            PoolMethodMapper={'0':'MAX', '1':'AVE', '2':'STOCHASTIC'}
            pooling_param_res.append('    pool: %s' % PoolMethodMapper[str(layer.pooling_param.pool)])
        if layer.pooling_param.kernel_h>0 and layer.pooling_param.kernel_w>0:
            PoolMethodMapper={'0':'MAX', '1':'AVE', '2':'STOCHASTIC'}
            pooling_param_res.append('    kernel_h: %d' % layer.pooling_param.kernel_h)
            pooling_param_res.append('    kernel_w: %d' % layer.pooling_param.kernel_w)
            pooling_param_res.append('    stride: %d' % layer.pooling_param.stride)
        if len(pooling_param_res)>0:
            res.append('  pooling_param {')
            res.extend(pooling_param_res)
            res.append('  }')
    # inner_product_param
    if layer.inner_product_param is not None:
        inner_product_param_res = list()
        if layer.inner_product_param.num_output!=0:
            inner_product_param_res.append('    num_output: %d' % layer.inner_product_param.num_output)
        if len(inner_product_param_res)>0:
            res.append('  inner_product_param {')
            res.extend(inner_product_param_res)
            res.append('  }')
    # drop_param
    if layer.dropout_param is not None:
        dropout_param_res = list()
        if layer.dropout_param.dropout_ratio!=0.5:
            dropout_param_res.append('    dropout_ratio: %f' % layer.dropout_param.dropout_ratio)
        if len(dropout_param_res)>0:
            res.append('  dropout_param {')
            res.extend(dropout_param_res)
            res.append('  }')
    res.append('}')
    for line in res:
        print line
