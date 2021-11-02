import cv2
import numpy as np

#img: grayscale uint8 image of the text-line to be segmented.
#kernelSize: size of filter kernel, must be an odd integer.
#sigma: standard deviation of Gaussian function used for filter kernel.
#theta: approximated width/height ratio of words, filter function is distorted by this factor.
#minArea: ignore word candidates smaller than specified area.

def wordSegmentation(img, kernelSize, sigma, theta, minArea=0):
    ''' word segmentation '''
    sigma_X = sigma
    sigma_Y = sigma * theta
    height, width = img.shape
    # use gaussian blur and applies threshold
    imgFiltered = cv2.GaussianBlur(img, (kernelSize, kernelSize), sigmaX=sigma_X, sigmaY=sigma_Y)#Kernel -10 de lay chieu doc nhieu hon
    #cv2.imshow ('imgFiltered',imgFiltered)
    _, imgThres = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgThres = 255 - imgThres
    #cv2.imshow('imgThresgeg',imgThres)
    # find connected components
    components, _ = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    items = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < minArea:
            continue
        # append bounding box and image of word to items list
        currBox = cv2.boundingRect(c)
        (x, y, w, h) = currBox
        
        if (w/h >= 3 and h < height/2 and y == 0): #Loại chữ môn thi nếu bị dính
            continue
        else:
            currImg = img[max(0, y-5):y+h, x:x+5+w] # +-5 de lay dau tieng Viet
            items.append([currBox, currImg])

    result = []
    temp = []
    for currBox, currImg in items:
        temp.append([currBox, currImg])
    # list of words, sorted by x-coordinate
    result.append(sorted(temp, key=lambda entry: entry[0][0]))
    return result

def main():
    name_crop = cv2.imread('doc/removelineresult/namecrop_giaythi12.jpg')
    name_crop = cv2.cvtColor(name_crop, cv2.COLOR_BGR2GRAY)
    cv2.imshow('name_crop',name_crop)
    print ('name_crop',name_crop.shape)

    result = wordSegmentation(name_crop, kernelSize=21, sigma=11, theta=4, minArea=500)
    draw = []
    i = 0
    for line in result:
        if len(line):
            for (_, w) in enumerate(line):
                (wordBox, wordImg) = w
                cv2.imshow('wordImg '+ str(i),wordImg)
                print ('wordImg',wordImg.shape)
                draw.append(wordBox)
                i = i+1
    for wordBox in draw:
        (x, y, w, h) = wordBox
        cv2.rectangle(name_crop, (x, y), (x+w, y+h), 0, 1)

    cv2.imshow('name_crop',name_crop)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

