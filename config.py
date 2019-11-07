#!/usr/bin/env python
# -*- coding:utf-8 -*-
#CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-~`<>'.:;^/|!?$%#@&*()[]{}_+=,\\\""
#NUM_CLASSES = len(CHAR_VECTOR) + 1

#CHAR_VECTOR = []
#NUM_CLASSES = 0

dict_file_path = 'E:/work/Item/OCR/FOTS_TF-dev/FOTS_TF-dev/dictionary/ICDAR_2019_dic.txt'
with open(dict_file_path, encoding="utf-8", mode='r') as f:
    chars = list(map(lambda char: char.strip('\r\n'), f.readlines()))
CHAR_VECTOR = chars
NUM_CLASSES = len(CHAR_VECTOR) + 1
