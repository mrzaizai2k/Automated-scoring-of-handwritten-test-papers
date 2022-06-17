
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

from helper import preprocess
from segment import wordSegmentation
from EAST import EASTimg, predictions
from Preprocessing import imformation_crop, removeline, removecircle
from digit_model import build_digit_model
from word_model import build_word_model
from Excel import class_list,lexicon_search,writing_to_excel

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

############# Initial Config

class_list_dir = 'data\Class_list.xlsx'
name_list, MSSV_list, name_MSSV_list, Diem_list = class_list(class_list_dir)

with open('data\scoreList.txt', "r", encoding="utf-8") as f:
    reader = f.read()
scoreDict = sorted(reader.split(' '))
print ('scoreList',scoreDict)

def num_to_label(num,alphabets):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

############## RESTORE MODEL ##########

#Name model
with open('data/charList.txt', 'r', encoding='utf-8') as f:
    alphabets_word = f.read()

max_str_len_word = 15
word_model, word_model_CTC = build_word_model(alphabets = alphabets_word, max_str_len = max_str_len_word)
#word_model.summary()
word_model_dir = 'model\model_word/2021-10-24\word_model_5_2021-10-24.h5'
#model\model_word/2021-10-09\word_model_last_6.h5
#model\model_word/2021-10-23\word_model_10_2021-10-23.h5
word_model.load_weights(word_model_dir)

## MSSV model
alphabets_digit = '0123456789'
max_str_len_digit = 10
digit_model, digit_model_CTC = build_digit_model(alphabets = alphabets_digit, max_str_len = max_str_len_digit)
#digit_model.summary()
digit_model_dir = 'model\multi_digit_model/2021-11-26_3\digit_model_last_2021-11-26.h5'
digit_model.load_weights(digit_model_dir)

    
start = time.time()
giaythi  = cv2.imread('data\Class_list_constrained\giaythi209.jpg', cv2.IMREAD_COLOR)
(MSSV_crop, name_crop, diem_crop) = imformation_crop(giaythi)

print ('\nimg shape',giaythi.shape)
print ('MSSV_crop shape',MSSV_crop.shape)
print ('diem_crop shape',diem_crop.shape)
print ('name_crop shape',name_crop.shape,'\n')

# cv2.imshow('diem_crop',diem_crop)
# cv2.imshow('MSSV_crop',MSSV_crop)
# cv2.imshow('name_crop',name_crop)


############### NAME_RECOGNITION ########
name_crop_copy = name_crop.copy()
name_crop_copy = removeline(name_crop_copy)
    
result = wordSegmentation(name_crop_copy, kernelSize=21, sigma=11, theta=4, minArea=500)
name_recognized = str()
draw = []
i = 0
for line in result:
    if len(line):
        for (_, w) in enumerate(line):
            (wordBox, wordImg) = w
            
            print ('wordImg.shape',wordImg.shape)
            #cv2.imshow('wordImg '+ str(i),wordImg)

            wordImg = preprocess(wordImg, imgSize = (128, 32))
            wordImg = np.array(wordImg).reshape(-1, 128, 32, 1)
            pred = word_model.predict(wordImg)
            
            decoded = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                        greedy=False,
                                        beam_width=50,
                                        top_paths=1)[0][0])
            
            name_recognized += num_to_label(decoded[0], alphabets = alphabets_word) + ' ' 
            #Its just an approx name
            draw.append(wordBox)
            i = i+1


for wordBox in draw:
        (x, y, w, h) = wordBox
        cv2.rectangle(name_crop, (x, y-3), (x+w+3, y+h), 0, 1)
#cv2.imshow('name_crop',name_crop)

############### MSSV_RECOGNITION #######################

#cv2.imshow('MSSV_crop',MSSV_crop)

MSSV_crop_copy = MSSV_crop.copy()

MSSV_crop_copy = removeline(MSSV_crop_copy)
print ('MSSV_crop_copy remove line',MSSV_crop_copy.shape)

# cv2.imshow('MSSV_crop_copy1',MSSV_crop_copy)

MSSV_crop_copy = preprocess(MSSV_crop_copy,(128,32))
MSSV_crop_copy = np.array(MSSV_crop_copy).reshape(-1, 128, 32, 1)


pred_MSSV = digit_model.predict(MSSV_crop_copy)

decoded_MSSV = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(pred_MSSV, input_length=np.ones(pred_MSSV.shape[0])*pred_MSSV.shape[1], 
                                        greedy=False,
                                        beam_width=5,
                                        top_paths=1)[0][0])

MSSV_recognized = num_to_label(decoded_MSSV[0], alphabets = alphabets_digit) 

print('\nNAME_approx: ' + name_recognized)
print('MSSV_approx: ' + MSSV_recognized)

name_MSSV_recognized = name_recognized.strip() + ' ' + MSSV_recognized.strip()        
name_MSSV_index, name_MSSV_recognized,_ = lexicon_search (name_MSSV_recognized, name_MSSV_list)
print ('\nname_MSSV_recognized:',name_MSSV_recognized)

############### DIEM_RECOGNITION #######################

diem_crop_copy = diem_crop.copy()
cv2.imshow('anh goc',diem_crop_copy)
diem_crop_copy = removecircle(diem_crop_copy)
print ('diem_crop_copy thresh hold',diem_crop_copy.shape)
#cv2.imshow('removecircle',diem_crop_copy)

diem_recognized =str()
diem_crop_copy = preprocess(diem_crop_copy, imgSize = (128, 32))
diem_crop_copy = np.array(diem_crop_copy).reshape(-1, 128, 32, 1)

# plt.figure(num='diem',figsize=(3,3))
# plt.imshow(np.squeeze(diem_crop_copy[0,:,:,]))
# plt.show()

pred_diem = digit_model.predict(diem_crop_copy)

decoded_diem = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(pred_diem, input_length=np.ones(pred_diem.shape[0])*pred_diem.shape[1], 
                                        greedy=False,
                                        beam_width=5,
                                        top_paths=1)[0][0])

diem_recognized = num_to_label(decoded_diem[0], alphabets = alphabets_digit) 
print ('diem approx', diem_recognized)
_, diem_recognized,_ = lexicon_search (diem_recognized, scoreDict)

if diem_recognized != '10':
    diem_recognized = diem_recognized[:1]+ '.' + diem_recognized[1:]
diem_recognized = float(diem_recognized)    

print('\ndiem_recognized: '+ str(diem_recognized)+'\n----------')
#writing_to_excel (class_list_dir, name_MSSV_index + 2,diem_recognized)

end = time.time()
# show timing information on text prediction
print("[INFO] main took {:.6f} seconds".format(end - start))

cv2.waitKey(0)

