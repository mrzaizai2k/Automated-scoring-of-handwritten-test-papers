##Loading the necessary packages
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt
import time

#Give location of the image to be read.
#"Example-images/ex24.jpg" image is being loaded here.


def EASTimg(image):

    #Saving a original image and shape
    orig = image.copy()
    (origH, origW) = image.shape[:2]
    print('origH',origH,'   origW',origW)
    #ratio = int((640*(origW/origH))/32)
    #print ('ratio',ratio)
    # set the new height and width to default 320 by using args #dictionary. 

    #(newW, newH) = (576,64)
    (newW, newH) = (1024,640)
    print('newH',newH,'   newW',newW)
    #Calculate the ratio between original and new image for both height and weight.

    #This ratio will be used to translate bounding box location on the original image.

    rW = origW / float(newW)
    rH = origH / float(newH)

    # resize the original image to new dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # construct a blob from the image to forward pass it to EAST model
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	    (123.68, 116.78, 103.94), swapRB=True, crop=False)

    #the mean values for the ImageNet training set are R=103.93, G=116.77, and B=123.68 (mean subtraction of blob)

    # load the pre-trained EAST model for text detection
    net = cv2.dnn.readNet('model/east_text_detection.pb')


    # We would like to get two outputs from the EAST model.
    #1. Probabilty scores for the region whether that contains text or not.
    #2. Geometry of the text -- Coordinates of the bounding box detecting a text
    # The following two layer need to pulled from EAST model for achieving this.

    layerNames = [

            "feature_fusion/Conv_7/Sigmoid",

            "feature_fusion/concat_3"]

    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    #Forward pass the blob from the image to get the desired output layers

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

        
    # Find predictions and  apply non-maxima suppression
    (boxes, confidence_val) = predictions(scores, geometry)
    boxes = non_max_suppression(np.array(boxes), probs=confidence_val)
    print('so luong boxes', len(boxes))
    for (startX, startY, endX, endY) in boxes:
	    # scale the bounding box coordinates based on the respective
	    # ratios
	    startX = int(startX * rW)
	    startY = int(startY * rH)
	    endX = int(endX * rW)
	    endY = int(endY * rH)
	    # draw the bounding box on the image
	    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    #print ('confi_val',confidence_val )
    #print('boxes',boxes )
    #cv2.putText(image, str(confidence_val), (257,71),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    #cv2.rectangle(image,(257,71),(288,85),(123,255,0),2)

    return orig

## Returns a bounding box and probability score if it is more than minimum confidence

def predictions(prob_score, geo):
    (numR, numC) = prob_score.shape[2:4]
    boxes = []
    confidence_val = []

    # loop over rows
    for y in range(0, numR):
            scoresData = prob_score[0, 0, y]
            x0 = geo[0, 0, y]
            x1 = geo[0, 1, y]
            x2 = geo[0, 2, y]
            x3 = geo[0, 3, y]
            anglesData = geo[0, 4, y]

            # loop over the number of columns
            for i in range(0, numC):
                if scoresData[i] < 0.5:
                        continue
                (offX, offY) = (i * 4.0, y * 4.0)

                # extracting the rotation angle for the prediction and computing the sine and cosine
                angle = anglesData[i]
                cos = np.cos(angle)
                sin = np.sin(angle)
                    
                # using the geo volume to get the dimensions of the bounding box
                h = x0[i] + x2[i]
                w = x1[i] + x3[i]

                # compute start and end for the text pred bbox
                endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
                endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
                startX = int(endX - w)
                startY = int(endY - h)

                boxes.append((startX, startY, endX, endY))
                confidence_val.append(scoresData[i])

        # return bounding boxes and associated confidence_val
    return (boxes, confidence_val)

def main():
    img = cv2.imread('data/sample/giaythi8.jpg', cv2.COLOR_BGR2GRAY)
    img = EASTimg(img)
    cv2.imshow ('EAST image',img)
    cv2.waitKey(0)

if __name__ =='__main__':
    main()