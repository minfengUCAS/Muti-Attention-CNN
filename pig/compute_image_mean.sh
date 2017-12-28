#!/usr/local/en sh
DATA=./data
/home/minfeng.zhan/workspace/release/caffe/build/tools/compute_image_mean $DATA/image_train_lmdb $DATA/image_mean_448_448.binaryproto
