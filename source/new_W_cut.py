import os
import time
import cv2
import numpy as np
from helper import preprocess
import datetime
import random

from word_model import build_word_model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


############# Initial Config


def num_to_label(num,alphabets):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

############## RESTORE MODEL ##########
def lineSegmentation(img, sigmaY):
    ''' line segmentation '''
    img = 255 - img
    Py = np.sum(img, axis=1)

    y = np.arange(img.shape[0])
    expTerm = np.exp(-y**2 / (2*sigmaY**2))
    yTerm = 1 / (np.sqrt(2*np.pi) * sigmaY)
    Gy = yTerm * expTerm

    Py_derivative = np.convolve(Py, Gy)
    thres = np.max(Py_derivative) // 2
    # find local maximum
    res = (np.diff(np.sign(np.diff(Py_derivative))) < 0).nonzero()[0] + 1

    lines = []
    for idx in res:
        if Py_derivative[idx] >= thres:
            lines.append(idx)
    return lines


def wordSegmentation(img, kernelSize, sigma, theta, minArea=0):
    ''' word segmentation '''
    sigma_X = sigma
    sigma_Y = sigma * theta
    # use gaussian blur and applies threshold
    imgFiltered = cv2.GaussianBlur(img, (kernelSize, kernelSize), sigmaX=sigma_X, sigmaY=sigma_Y)
    cv2.imwrite('data/real_word_data2/imgFiltered.png',imgFiltered)
    _, imgThres = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,7))
    imgThres = cv2.erode(imgThres,kernel,iterations=6)
    imgThres = 255 - imgThres
    cv2.imwrite('data/real_word_data2/imgThres.png',imgThres)
    # find connected components
    components, _ = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lines = lineSegmentation(img, sigma)

    items = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < minArea or cv2.contourArea(c) > 60000:
            continue

        # append bounding box and image of word to items list
        currBox = cv2.boundingRect(c)
        (x, y, w, h) = currBox
        if (0.8<= w/h <= 4):
            currImg = img[y:y+h, x:x+w]
            items.append([currBox, currImg])
    
    result = []
    for line in lines:
        temp = []
        for currBox, currImg in items:
            if currBox[1] < line:
                temp.append([currBox, currImg])
        for element in temp:
            items.remove(element)
        # list of words, sorted by x-coordinate
        result.append(sorted(temp, key=lambda entry: entry[0][0]))
    return result

def prepareImg(img, height):
    ''' convert given image to grayscale image (if needed) and resize to desired height '''
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)
#Name model
alphabets_word = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvxyzÀÁÂÔÚÝàáâãèéêìíòóôõùúýĂăĐđĩũƒƠơƯưạẢảẤấẦầẩẫậắằẳẵặẹẻẽếỀềỂểễỆệỉịọỏỐốỒồổỗộớờỞởỡợụỦủứừửữựỳỷỹ'

max_str_len_word = 15
word_model, word_model_CTC = build_word_model(alphabets = alphabets_word, max_str_len = max_str_len_word)
#word_model.summary()
word_model_dir = 'model\model_word/2021-10-24\word_model_last.h5'
#model\model_word/2021-10-09\word_model_last_6.h5
word_model.load_weights(word_model_dir)


for filename in os.listdir("data\Word_images2"):
 
    print ('\n................................\n')
    print ('filename',filename)
    filepath  = os.path.join("data\Word_images2/", filename)
    
    start = time.time()
    giaythi  = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
 
    ############### NAME_RECOGNITION ######################
   
    giaythi = 255 - cv2.threshold(giaythi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    result = wordSegmentation(giaythi, kernelSize=21, sigma=15, theta=4, minArea=6000)
    name_recognized = str()
    draw = []
    i = 0
    for line in result:
        if len(line):
            for (_, w) in enumerate(line):
                (wordBox, wordImg) = w
                #print ('wordImg.shape',wordImg.shape)
                #cv2.imshow('wordImg '+ str(i),wordImg)
                img = wordImg
                wordImg = preprocess(wordImg, imgSize = (128, 32))
                wordImg = np.array(wordImg).reshape(-1, 128, 32, 1)
                pred_names = word_model.predict(wordImg)

                decoded_names = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(pred_names, input_length=np.ones(pred_names.shape[0])*pred_names.shape[1], 
                                        greedy=False,
                                        beam_width=50,
                                        top_paths=1)[0][0])
                name_recognized = num_to_label(decoded_names[0], alphabets = alphabets_word) 
                cv2.imwrite('data/real_word_data2/'+ str(name_recognized.strip())+'.png', img)
                #Its just an approx name
                draw.append(wordBox)
                i = i+1

    for wordBox in draw:
            (x, y, w, h) = wordBox
            cv2.rectangle(giaythi, (x, y-3), (x+w+3, y+h), 0, 1)
    cv2.imshow('giaythi',giaythi)

cv2.waitKey(0)
