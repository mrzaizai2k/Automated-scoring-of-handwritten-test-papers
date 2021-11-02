
import os
import cv2
import csv
import random
from random import choice 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import tensorflow as tf
from tensorflow import keras

from helper import preprocess

########## Initial config ##############
header = 'label image'
header = header.split()
with open('data/aug_digit_data.csv', "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)


############# Main##############
def elastic_transform(image, alpha, sigma, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    dx = gaussian_filter((random_state.rand(h,w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(h,w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    
    return distored_image.reshape(image.shape)

def number_augment(image):
    ''' The dataset is so clean, so it's not fit with out real input images with blobs, noise, ugly characters '''
    height, width = image.shape 
    
    # dilate/erode image
    method = random.randint(0,2)

    if method == 1:
        kernelsize = [(1,2), (2,1), (2,2), (3,2), (2,3), (3,3),(5,5), (8,2), (2,8)]
        #print ('kernel size', random.sample(kernelsize,1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, random.sample(kernelsize,1)[0])
        image = cv2.erode(image, kernel)
    elif method == 2:
        kernelsize = [(1,2), (2,1), (2,2), (3,2), (2,3)]
        #print ('kernel size', random.sample(kernelsize,1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, random.sample(kernelsize,1)[0])
        image = cv2.dilate(image, kernel)

    # add blob noise 
    num_blob= int(random.gauss(10,10))
    for i in range (num_blob):
        x = random.randint(0,width)
        y = int (random.gauss(0,height))
        radius = random.randint(0,8)
        
        image = cv2.circle(image, (x,y), radius = radius, color = ((0, 0, 0) ), thickness = -1) 
    
    # #add circle noise
    # if digits_per_sequence in [1,3,4]:
    #     num_circle= random.randint(0,1)
    #     for i in range (num_circle):
    #         x = random.randint(0,width)
    #         y = random.randint(12,16)
    #         radius = random.randint(13,15)
    #         image = cv2.circle(image, (x,y), radius = radius, color = ((0, 0, 0) ), thickness = 1)
    
    #add line noise
    num_line = random.randint(0,1)
    for i in range(num_line):
        p1 = (random.randint(0,int(width/2)), random.randint(int(height/2),height)) # (x,y)
        p2 = (random.randint(int(width/2),width), random.randint(int(height/2),height)) # (x,y)
        cv2.line(image, p1, p2, (0, 0, 0), thickness = 1)

    #Random vertical cut out
    num_cut = random.randint(0,5)
    for i in range(num_cut):
        p1 = (random.randint(0,width), 0) # (x,y)
        p2 = (p1[0]+1, p1[1]+32) # (x,y)
        cv2.rectangle(image, p1, p2, (255,255,255), -2)

    #Random horizontal cut out
    num_line = random.randint(0,1)
    for i in range(num_line):
        p1 = (random.randint(0,int(width/2)), random.randint(int(height/2),height)) # (x,y)
        p2 = (random.randint(int(width/2),width), random.randint(int(height/2),height)) # (x,y)
        cv2.line(image, p1, p2, (255, 255, 255), thickness = 1)

    #Elastic transform
    chance  = random.randint(0,100) 
    if chance < 10: # Chỉ 10% data được elastic transform
        image  =  elastic_transform (image, image.shape[1] * 2, image.shape[1] * 0.08)

    return image

c=0
for filename in os.listdir("data\Multi_digit_data/real_num_data"):
    filepath  = os.path.join("data\Multi_digit_data/real_num_data/", filename)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    label ="'" +str( filename[:-4])
    img = 255 - cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    for i in range (1):
        image = number_augment(img)
        #cv2.imwrite('data/'+str(label)+'.png',image)
        image = preprocess(image, imgSize = (128, 32))

        # Translate image
        tx = random.uniform(-5, 5)
        ty = 0
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty]], dtype=np.float32)
        image = cv2.warpAffine(src=image, M=translation_matrix, 
                                            dsize=(image.shape[1], image.shape[0]))
                                                    
        # Rotating the image     
        center = (image.shape[1]//2, image.shape[0]//2)
        angle = random.randint (-10,10)
        scale = random.uniform(0.6,0.8)
        rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
        image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))

        image = image.flatten()
        #print ('set', set(image.flatten()))
        image = ( " ".join( str(e) for e in image ) )
        value = [label, image]

        with open('data/aug_digit_data12.csv', "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(value)
    
    print ('processing {:.2f}%'.format(c*100/225))
    c+=1

print('\ndone')


############# For testing ###########

data = pd.read_csv('data/aug_digit_data.csv')

labels = data.iloc[: ,0].to_list()
images = data.iloc[: ,1].to_list()
print ('len', len (labels))
t = []
for image in images:
    image = image.split(' ')
    image = np.array(image, dtype = float)
    t.append(image)
images = t
t = []
images = np.array(images).reshape(-1, 128, 32, 1)

for label in labels:
    label = label.split("'")[1]
    t.append(label)
labels = t
t = []

MSSV_crop_copy = cv2.imread('doc\MSSV_input\MSSVcrop_giaythi9.jpg', cv2.IMREAD_GRAYSCALE)
MSSV_crop_copy = preprocess(MSSV_crop_copy,(128,32))
MSSV_crop_copy = np.array(MSSV_crop_copy).reshape(-1, 128, 32, 1)

plt.figure(num='compare MSSV n multi digit',figsize=(9,9))
for i in range(10):

    plt.subplot(2,5,i+1) 
    rand = random.randint(0,400)
    plt.title(labels[rand])
    plt.imshow(np.squeeze(images[rand,:,:,]))
    
    #plt.axis('off')
plt.title('1710586')
plt.imshow(np.squeeze(MSSV_crop_copy[0,:,:,]))    
plt.show()

cv2.waitKey(0)



