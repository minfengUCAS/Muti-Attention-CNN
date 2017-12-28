#! /usr/bin/env python
#-*- coding:utf-8 -*-

import os
import sys
import glob

# 生成图片列表文件
def createFileList(image_path, txt_path):
    fw = open(txt_path, "w")
    
    class_names = os.listdir(image_path)
    class_names.sort()

    for cls in class_names:
        second_file_pattern = os.path.abspath(os.path.join(image_path, cls))+"/*.jpg"
        print second_file_pattern
        image_file_list = glob.glob(second_file_pattern)
        for image_file in image_file_list:
            res = ""
            for i in range(1,7):
                image_file = image_file[0:46]+str(i)+image_file[47:] 
                res += image_file+"\t"
            print res+cls+"\n"
            fw.write(res+cls+"\n")

    fw.close()

    print("生成txt文件成功！")

    fw.close()

if __name__=="__main__":
    image_path = r"./data/1"
    #image_path = r"./test"
    #txt_path = r"./train_list.txt"
    txt_path = r"./cls_train_list.txt"
    createFileList(image_path, txt_path)
