# -*- coding: utf-8 -*-
# @Time    : 2019/5/24 21:52
# @Author  : YuYi
txt_path = 'G:/extra_work/Multi_label_Image_Classification/data/train2014.txt'
import os
import numpy as np
import cv2
import sys
sys.path.append('../')
import args


def one_hot_multi_label(labels, dim=20):
    '''
    :param labels: list
    :return:
    out_array:shape:[dim]
    '''
    out_array = np.zeros([dim], dtype=np.float32)
    out_array[labels] = 1
    return out_array


def parse_line(line):
    '''
    Given a line from the training/test txt file, return image and onehot_label
    return:
        pic_path: string.
        labels: shape [-1]. class index.
    '''
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip('\n').split('\t')
    pic_name = s[0]
    pic_path = os.path.join(args.train_image_path, pic_name)
    img = cv2.imread(pic_path)
    img = cv2.resize(img, (args.image_w, args.image_h), interpolation=0)
    img = img.astype(np.float32)
    labels = [int(i) for i in s[1].split(',')]
    onehot_labels = one_hot_multi_label(labels)
    return img, onehot_labels


if __name__ == '__main__':
    print(args.train_txt_path)
