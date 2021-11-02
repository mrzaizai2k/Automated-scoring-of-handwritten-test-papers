
import cv2
import os
import numpy as np 
import pandas as pd 
import datetime
import random

import subprocess
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold, cross_val_score, KFold,StratifiedKFold 

from pathlib import Path
from collections import Counter

from helper import preprocess
from digit_model import build_digit_model

datetime_object = datetime.date.today()
if not os.path.exists('model\multi_digit_model/'+ str(datetime_object)):
    os.mkdir('model\multi_digit_model/'+ str(datetime_object))

########## LOAD DATA ######################################################
data = pd.read_csv('data/aug_digit_data1.csv')

labels = data.iloc[:100 ,0].to_list() #if you want to read 100 images use labels = data.iloc[:100 ,0].to_list(
images = data.iloc[:100 ,1].to_list()

print ('\nThe number of data', len(labels))
# Preprocess data and delete false data
count = 0
for i, label in enumerate(labels):
    if type(label) == float:
        del labels[i]
        del images[i]
        count +=1
print ('The number of invalid data:',count)
print ('The number of valid data left:',len(labels))
##

t = []
for i,image in enumerate(images):
    image = image.split(' ')
    image = np.array(image, dtype = float)
    t.append(image)

images = t
t=[]
print ('len images', len(images))

for label in labels:
    label = label.split("'")[1]
    t.append(label)
labels = t
t = []

images = np.array(images).reshape(-1, 128, 32, 1)

############### SPLIT DATA INTO TRAIN_VALID SET ###########

X_train, X_valid, y_train, y_valid = train_test_split(images, labels, train_size= 0.8, shuffle = True)

print ('\nlen(X_train)',len(X_train))
print ('len(X_valid)',len(X_valid))
print ('\n X_train.shape',X_train.shape)
print ('\n X_valid.shape',X_valid.shape)


# plt.figure(num='char',figsize=(9,18))
# for i in range(6):
#     rand = random.randint(0, len(X_train))
#     plt.subplot(3,3,i+1) 
#     plt.title(y_train[rand])
#     plt.imshow(np.squeeze(X_train[rand,:,:,]))
#     plt.axis('off')
# plt.show()

alphabets = "0123456789"
max_str_len = 10 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 31 # max length of predicted labels


def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret


#train_y contains the true labels converted to numbers and padded with -1. 
#The length of each label is equal to max_str_len.
#train_label_len contains the length of each true label (without padding)
#train_input_len contains the length of each predicted label. 
#The length of all the predicted labels is constant i.e number of timestamps - 2.
#train_output is a dummy output for ctc loss.

train_y = np.ones([len(X_train), max_str_len]) * -1
train_label_len = np.zeros([len(X_train), 1])
train_input_len = np.ones([len(X_train), 1]) * (num_of_timestamps-2)
train_output = np.zeros([len(X_train)])

for i in range(len(X_train)):
    train_label_len[i] = len(y_train[i])
    train_y[i, 0:len(y_train[i])]= label_to_num(y_train[i])  
    

valid_y = np.ones([len(X_valid), max_str_len]) * -1
valid_label_len = np.zeros([len(X_valid), 1])
valid_input_len = np.ones([len(X_valid), 1]) * (num_of_timestamps-2)
valid_output = np.zeros([len(X_valid)])

for i in range(len(X_valid)):
    valid_label_len[i] = len(y_valid[i])
    valid_y[i, 0:len(y_valid[i])]= label_to_num(y_valid[i])  

#print('\n True y_train  : ',y_train[10] , '\ntrain_y : ',train_y[10],'\ntrain_label_len : ',train_label_len[10], '\ntrain_input_len : ', train_input_len[10])


digit_model, digit_model_CTC = build_digit_model(alphabets = alphabets, max_str_len = max_str_len )
#digit_model.summary()
#model_final.summary()

epochs = 10
batch_size = 32
early_stopping_patience = 10

def scheduler(epoch):
    if epoch <= 20:
        return 1e-3  
    elif 20 < epoch <= 25:
        return 1e-4
    else:
        return 1e-5
# Add early stopping

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='model/multi_digit_model/'+str(datetime_object)+'/digit_model_{epoch:02d}_test.h5', 
                                       save_freq='epoch',
                                       monitor='val_loss',
                                       mode='min', 
                                       save_best_only=True,
                                       period = 10),
    tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )
]

digit_model_CTC.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, 
                    optimizer=keras.optimizers.Adam(),                  
                    )

history = digit_model_CTC.fit(x=[X_train, train_y, train_input_len, train_label_len], y=train_output, 
                validation_data=([X_valid, valid_y, valid_input_len, valid_label_len], valid_output),
                epochs = epochs, 
                batch_size = batch_size,
                callbacks = my_callbacks,
                )

# list all data in history
print(history.history.keys())


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model/multi_digit_model/'+str(datetime_object)+'/digit_model_loss_test.png')
plt.show()

with open('model/multi_digit_model/'+str(datetime_object)+'/digit_model_test.txt', 'w', encoding='utf-8') as f:
    f.write('len(X_train): {} \nlen (X_valid): {} \n'.format(len(X_train), len (X_valid)))
    f.write('max_str_len: {} \nnum_of_characters: {} \nnum_of_timestamps: {} \n'.format(max_str_len,num_of_characters,num_of_timestamps))
    f.write('batch_size: {} \nepochs: {} \n'.format(batch_size,epochs) )

############ test #######

preds = digit_model.predict(X_valid)
#print('\n preds',preds)
decoded = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                    greedy=True,)[0][0])
#print ('\n decoded',decoded)

prediction = []
for i in range(len(X_valid)):
    prediction.append(num_to_label(decoded[i]))
    
print ('\n predict',num_to_label(decoded[0]))

y_true = y_valid
correct_char = 0
total_char = 0
correct = 0

for i in range(len(X_valid)):
    pr = prediction[i]
    tr = y_true[i]
    total_char += len(tr)
    
    for j in range(min(len(tr), len(pr))):
        if tr[j] == pr[j]:
            correct_char += 1
            
    if pr == tr :
        correct += 1 
    
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('Correct words predicted      : %.2f%%' %(correct*100/len(X_valid)))

with open('model/multi_digit_model/'+str(datetime_object)+'/digit_model_test.txt', 'a', encoding='utf-8') as f:
    f.write('\nCorrect characters predicted : %.2f%%' %(correct_char*100/total_char))
    f.write('\nCorrect words predicted      : %.2f%%' %(correct*100/len(X_valid)))

cv2.waitKey(0)