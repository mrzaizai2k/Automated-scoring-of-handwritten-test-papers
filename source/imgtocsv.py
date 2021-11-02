#Convert Image into csv file to reduce the size of dataset
# It's much convenient when you want to upload and train it on Kaggle 

import os 
import cv2
import numpy as np 
import pandas as pd
import csv
import random

from sklearn.utils import shuffle
from helper import preprocess
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold, cross_val_score, KFold,StratifiedKFold 
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class imgtocsv:
    def __init__(self, dataPath, imgSize):
        ''' loader for dataset at given location '''

        self.dataPath = dataPath
        
        self.imgSize = imgSize
        self.images = []
        self.labels = []
        self.chars = set()

    
    def load_char_img(self):

        print ('dataPath',self.dataPath)
        print ('The number of image folder: ', len(os.listdir(self.dataPath)))
        i=0
        for i,folderName in enumerate(os.listdir(self.dataPath)): 
            #dataPath: CRNN\InkData_word_processed
            #folderName: 20140603_0003_KQBDVN
            baseLabels = os.path.join(self.dataPath, folderName + '/labels/') #tao duong dan den folder moi
            baseImages = os.path.join(self.dataPath, folderName + '/images/')
            
            i += 1
            print('\n---------Processing: %d/255---------' %i)

            for fileName in os.listdir(baseLabels):

                filePath = os.path.join(baseLabels, fileName)
                # get ground truth text
                with open(filePath, 'r', encoding='utf-8') as f:
                    gtText = f.read()
                # get image file path
                image = os.path.join(baseImages, fileName.split('.')[0] + '.png')
    
                image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                image = augmentation(image)
                image = preprocess(image, self.imgSize)
                                                            
                # Rotating the image     
                center = (image.shape[1]//2, image.shape[0]//2)
                angle = random.randint (-10,10)
                scale = random.uniform(0.8,1.2)
                rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
                image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))

                image = image.flatten()
                #print ('set', set(image.flatten()))
                image = ( " ".join( str(e) for e in image ) )
                value = [gtText, image]
                # get all characters in dataset
                self.chars = self.chars.union(set(list(gtText)))
                # put sample into list

                with open('data/aug_word_data.csv', "a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(value)

            if i == 1: #Nếu muốn đọc hết folder thì #
                break

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

def augmentation(image):
    ''' The dataset is so clean, so it's not fit with out real input images with blobs, noise, ugly characters '''
    height, width = image.shape 
    
    # dilate/erode image
    method = random.randint(0,2)
    kernelsize = [(5,1), (1,5), (3,3), (5,5), (8,2), (2,8), (1,10)]
    #print ('kernel size', random.sample(kernelsize,1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, random.sample(kernelsize,1)[0])
    if method == 1:
        image = cv2.erode(image, kernel)
    elif method == 2:
        image = cv2.dilate(image, kernel)

    # add blob noise 
    num_blob= int(random.gauss(30,30))
    for i in range (num_blob):
        x = random.randint(0,width)
        y = int (random.gauss(0,height))
        radius = random.randint(0,15)
        
        image = cv2.circle(image, (x,y), radius = radius, color = ((0, 0, 0) ), thickness = -1) 
    
    #add line noise
    num_line = random.randint(0,1)
    for i in range(num_line):
        p1 = (random.randint(0,15), random.randint(height-40,height)) # (x,y)
        p2 = (random.randint(width-15,width), random.randint(height-40,height)) # (x,y)
        cv2.line(image, p1, p2, (0, 0, 0), thickness = random.randint(5,15))

    #Random vertical cut out
    num_cut = random.randint(0,5)
    for i in range(num_cut):
        p1 = (random.randint(0,width), 0) # (x,y)
        p2 = (p1[0]+10, p1[1]+300) # (x,y)
        cv2.rectangle(image, p1, p2, (255,255,255), -2)

    #Random horizontal cut out
    num_line = random.randint(0,1)
    for i in range(num_line):
        p1 = (random.randint(0,15), random.randint(height-40,height)) # (x,y)
        p2 = (random.randint(width-15,width), random.randint(height-40,height)) # (x,y)
        cv2.line(image, p1, p2, (255, 255, 255), thickness = random.randint(15,25))

    #Elastic transform
    chance  = random.randint(0,100) 
    if chance < 10: # Chỉ 10% data được elastic transform
        image  =  elastic_transform (image, image.shape[1] * 2, image.shape[1] * 0.08)

    return image

def main():
    header = 'label image'
    header = header.split()
    print (header)

    with open('data/aug_word_data.csv', "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    convert = imgtocsv('InkData_word_processed', imgSize = (128, 32))
    convert.load_char_img()


    data = pd.read_csv('data/aug_word_data.csv')
    
    labels = data.iloc[:100 ,0].to_list()
    images = data.iloc[:100 ,1].to_list()
    t = []
    for image in images:
        image = image.split(' ')
        image = np.array(image, dtype = float)
        t.append(image)
    images = t
    images = np.array(images).reshape(-1, 128, 32, 1)

    plt.figure(num='char',figsize=(9,18))
    for i in range(9):
        plt.subplot(3,3,i+1) 
        plt.title(labels[i])
        plt.imshow(np.squeeze(images[i,:,:,]))
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()