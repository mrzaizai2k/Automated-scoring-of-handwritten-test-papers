#Resize, transpose and standardization grayscale images
#Image Augmentation

import os
import numpy as np 
import cv2 
import random

def preprocess(img, imgSize):
    ''' resize, transpose and standardization grayscale images '''
    # create target image and copy sample image into it
    
    widthTarget, heightTarget = imgSize 
    height, width = img.shape 
    factor_x = width / widthTarget
    factor_y = height / heightTarget
    factor = max(factor_x, factor_y)
    # scale according to factor
    newSize = (min(widthTarget, int(width / factor)), min(heightTarget, int(height / factor))) 
    #print ('newSize ={}, old size = {}'.format(newSize, img.shape ))
    img = cv2.resize(img, newSize)
    target = np.ones(shape=(heightTarget, widthTarget), dtype='uint8') * 255 #tao ma tran 255 (128,32)
    target[0:newSize[1], 0:newSize[0]] = img #Padding trên hoặc dưới

    #transpose
    img = cv2.transpose(target)
    # standardization
    mean, stddev = cv2.meanStdDev(img)
    mean = mean[0][0]
    stddev = stddev[0][0] # standard deviation
    #print ('mean ={}, stddev = {}'.format(mean, stddev))
    img = img - mean    
    img = img // stddev if stddev > 0 else img 
    #print ('set', set(img.flatten()))
    #img out co shape (128,32)
    return img


def main():
    img = cv2.imread('data\Multi_digit_data\multi_digit_images_test/0633054.png', cv2.IMREAD_GRAYSCALE)
    img = preprocess(img, imgSize=(128,32))

    cv2.waitKey(0)

if __name__ == '__main__':
    main()