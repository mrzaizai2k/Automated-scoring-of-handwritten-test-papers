import os
import time
import cv2
import numpy as np
from helper import preprocess
import datetime
import random

from segment import wordSegmentation
from EAST import EASTimg, predictions
from Preprocessing import imformation_crop, removeline, removecircle
from digit_model import build_digit_model
from word_model import build_word_model
from create_metrics_OCR import cer, wer, _levenshtein_distance
from Excel import class_list,lexicon_search


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


############# Initial Config ###################

class_list_dir = 'data\Class_list.xlsx'
name_list, MSSV_list, name_MSSV_list = class_list(class_list_dir)

def num_to_label(num,alphabets):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

image_size = 0
acc = 0
uncorrect = []
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
digit_model_dir = 'model\multi_digit_model/2021-10-30\digit_model_last_2021-10-30.h5'
#model\multi_digit_model/2021-10-20\digit_model_last_2021-10-20.h5
#model\multi_digit_model/2021-10-23\digit_model_last_2021-10-23_2.h5
digit_model.load_weights(digit_model_dir)

for filename in os.listdir("data\Class_list"):
    image_size += 1
    print ('\n................................\n')
    print ('filename',filename)

    real_index = int(filename[7:-4])

    filepath  = os.path.join("data\Class_list"+ '/', filename)
    
    start = time.time()
    giaythi  = cv2.imread(filepath, cv2.IMREAD_COLOR)
    (MSSV_crop, name_crop, diem_crop) = imformation_crop(giaythi)

    # print ('\nimg shape',giaythi.shape)
    # print ('MSSV_crop shape',MSSV_crop.shape)
    # print ('diem_crop shape',diem_crop.shape)
    # print ('name_crop shape',name_crop.shape,'\n')

    ############### NAME_RECOGNITION ######################
    name_crop_copy = name_crop.copy()
    name_crop_copy = removeline(name_crop_copy)

    #if not os.path.exists('doc/removelineresult'):
    #    os.mkdir('doc/removelineresult')
    #imwritepath = os.path.join('doc/removelineresult/namecrop_' + filename)
    #cv2.imwrite(str(imwritepath),name_crop_copy)
    
    result = wordSegmentation(name_crop_copy, kernelSize=21, sigma=11, theta=4, minArea=500)
    name_recognized = str()
    draw = []
    i = 0
    for line in result:
        if len(line):
            for (_, w) in enumerate(line):
                (wordBox, wordImg) = w
                #print ('wordImg.shape',wordImg.shape)
                #cv2.imshow('wordImg '+ str(i),wordImg)

                wordImg = preprocess(wordImg, imgSize = (128, 32))
                wordImg = np.array(wordImg).reshape(-1, 128, 32, 1)
                pred_names = word_model.predict(wordImg)

                decoded_names = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(pred_names, input_length=np.ones(pred_names.shape[0])*pred_names.shape[1], 
                                        greedy=False,
                                        beam_width=50,
                                        top_paths=1)[0][0])
                name_recognized += num_to_label(decoded_names[0], alphabets = alphabets_word) + ' ' 
                #Its just an approx name
                draw.append(wordBox)
                i = i+1
    

    ############### MSSV_RECOGNITION #######################

    #cv2.imshow('MSSV_crop',MSSV_crop)

    MSSV_crop_copy = MSSV_crop.copy()
    MSSV_crop_copy = removeline(MSSV_crop_copy)
    
    #cv2.imshow('MSSV_crop_copy',MSSV_crop_copy)

    MSSV_crop_copy = preprocess(MSSV_crop_copy,(128,32))

    pred_MSSV = digit_model.predict(np.array(MSSV_crop_copy).reshape(-1, 128, 32, 1))
    decoded_MSSV = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(pred_MSSV, input_length=np.ones(pred_MSSV.shape[0])*pred_MSSV.shape[1], 
                                        greedy=False,
                                        beam_width=5,
                                        top_paths=1)[0][0])

    MSSV_recognized = num_to_label(decoded_MSSV[0], alphabets = alphabets_digit) 
    
    print('\nNAME_approx: ' + name_recognized)
    print('MSSV_approx: ' + MSSV_recognized)

    name_MSSV_recognized = name_recognized.strip() + ' ' + MSSV_recognized.strip()        
    name_MSSV_index, name_MSSV_recognized = lexicon_search (name_MSSV_recognized, name_MSSV_list)
    print ('\nname_MSSV_recognized:',name_MSSV_recognized)

    if real_index == name_MSSV_index + 2:
        acc += 1    
        print ('Name & MSSV Accuracy:%.2f%%' %(acc*100/image_size))
    else:
        uncorrect.append(real_index)
    ############### Diem_RECOGNITION #######################

    diem_crop_copy = diem_crop.copy()

    diem_crop_copy = removecircle(diem_crop_copy)

    diem_crop_copy = preprocess(diem_crop_copy,(128,32))
    diem_crop_copy = np.array(diem_crop_copy).reshape(-1, 128, 32, 1)

    pred_diem = digit_model.predict(diem_crop_copy)
    decoded_diem = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(pred_diem, input_length=np.ones(pred_diem.shape[0])*pred_diem.shape[1], 
                                            greedy=False,
                                            beam_width=5,
                                            top_paths=1)[0][0])

    diem_recognized = num_to_label(decoded_diem[0], alphabets = alphabets_digit) 
    pr = diem_recognized.strip() #strip xóa kí tự trắng 2 đầu
    tr = '9'

    print('\ndiem_recognized: '+ diem_recognized+'\n----------')
    end = time.time()
    # show timing information on text prediction
    print("[INFO] main took {:.6f} seconds".format(end - start))


print ('\nThe number of correct recognization: {}. The total images: {}'.format(acc, image_size))
print ('The uncorrect pictures are:', uncorrect)
print ('Name & MSSV Accuracy:%.2f%%' %(acc*100/image_size))
cv2.waitKey(0)
