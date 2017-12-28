#!/usr/bin/en sh
DATA=./data
rm -rf $DATA/image_train_lmdb
/home/minfeng.zhan/workspace/release/caffe/build/tools/convert_imageset --shuffle /home/minfeng.zhan/dataset/Pig/train/ $DATA/train.txt $DATA/image_train_lmdb


