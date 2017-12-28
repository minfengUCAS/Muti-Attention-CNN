# -*- coding: utf-8 -*-

import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

import numpy as np
import tempfile
import time

from caffe.proto import caffe_pb2

def solver_deploy(train_net_path, test_net_path=None, solver_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path

    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000 # Test after every 1000 training iterations
        s.test_iter.append(100) # Test on 100 batches each time we test

    # The number of iterations over which to average the gradient
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization
    s.iter_size = 4

    s.max_iter = 100000 # of times to update the net (training iterations)

    # Solve using the stochastic gradient decent (SGD) algorithm.
    # Other choices include 'Adam' adn 'RMSProp'
    s.type = 'SGD'

    # Set the initial learning rate for SGD
    s.base_lr = base_lr

    # Set 'lr_policy' to define how the learning rate changes during traing.
    # Here, we 'step' the learning rate by mulitplying it by a factor 'gamma'
    # every 'stepsize' iteration
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 1000

    # Set other SGD hyperparameters. Setting a nonzero 'momentum' takes a 
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations
    s.display = 1000

    # Snapshots are files used to store networks we've trained. Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 200
    s.snapshot_prefix = '/home/minfeng.zhan/workspace/release/pig/model_cls/finetune_pig_on_bird'

    #  Train on the GPU. 
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename
    if solver_path is None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(str(s))
            return f.name
    else:
        with open(solver_path, 'w') as f:
            f.write(str(s))
            return solver_path

def run_solvers(niter, solvers, weights_dir=None, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""

    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name,_ in solvers}
                for _ in blobs)

    for it in range(niter):
        for name, s in solvers:
            s.step(1) # run a single SGD in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                            for b in blobs)
        if it % disp_interval == 0 or it+1 == niter:
            loss_disp = ':'.join('%s: loss=%.3f, acc=%2d%%' %
                                (n, loss[n][it], np.round(100*acc[n][it]))
                                for n,_ in solvers)
            timestemp = time.strftime('%Y-%m-%d %H:%M:%S' , time.localtime())
            print '%3d) %s %s' % (it, timestemp, loss_disp)

    # Save the learned weights from both nets
    if weight_dir is None:
        weight_dir = tempfile.mkdtemp()

    weights = {}

    for name, s in slovers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights


def main():
    niter = 10000
    base_lr = 0.00001
    train_net_path = r'/home/minfeng.zhan/workspace/release/pig/deploy/pig_cls_deploy_train.prototxt'
    solver_path = r'./data/pig_cls_solver.prototxt'
    weights_path = r'/home/minfeng.zhan/workspace/release/model/finetune_pig_on_bird_iter_200.caffemodel'
    weights_dir = r'/home/minfeng.zhan/workspace/release/pig/model_cls/'

    solver_filename = solver_deploy(train_net_path, solver_path=solver_path, base_lr=base_lr)
    solver = caffe.get_solver(solver_path)
    solver.net.copy_from(weights_path)

    print 'Runing solvers for %d iterations...' % niter
    solvers = [('finetunePigonBird', solver)]
    _,_, finetuned_weights = run_solvers(niter, solvers, weights_dir)
    print 'Done.'

if __name__ == '__main__':
    main()
