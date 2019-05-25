# -*- coding: utf-8 -*-
# @Time    : 2019/5/25 10:43
# @Author  : YuYi
import random
import sys
sys.path.append('../')
import args

'''
split train data set to train data set and train_val data set.
train data set is for training model and train_val data set is for test the model's performance
'''


def main():
    all_data = open(args.train_txt_path).readlines()
    random.shuffle(all_data)
    train_data = all_data[:args.num_train_img]
    train_val_data = all_data[args.num_train_img:]
    write_train_data = open(args.train_dataset_files, 'w')
    write_train_data.writelines(train_data)
    write_train_data.close()
    write_trainval_data = open(args.trainval_dataset_files, 'w')
    write_trainval_data.writelines(train_val_data)
    write_trainval_data.close()


if __name__ == '__main__':
    main()


