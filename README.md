# Information-recognition-on-the-university-test-paper
## Table of contents
* [1. Introduction](#1-Introduction)
* [2. Dataset](#2-Dataset)
* [3. Image preprocessing](#3-Image-preprocessing)
* [4. Word segmentation](#4-Word-segmentation)
* [4. Imbalanced data](#4-Imbalanced-data)
* [5. Model](#5-Model)
* [6. K - fold cross validation](#6-K-fold-cross-validation)

## 1. Introduction
This project is a part of my thesis. In short, you guys may or may not know that our teachers spend too much time on updating scores. Around 4000 test papers/a year on average for a secondary school teacher according to this [news](https://giaoduc.net.vn/giao-duc-24h/thong-tu-so-26-2020-tt-bgddt-da-go-bo-duoc-nhieu-ap-luc-cho-hoc-tro-va-giao-vien-post212222.gd)

The whole thesis is to help teacher update score into Excel automatically after writing score on test paper. But in this project we just extract infomation such as names, student ID, and recognize them have index to update score 

<p align="center"><img src="data/sample/giaythi5.jpg" width="500"></p>
<p align="center"><i>Hình 1. Test paper of Ho Chi Minh University of Technology </i></p>

As you can see, I use my university's test paper. My name is Mai Chi Bao and my student ID (MSSV) is 1710586. Those are handwritten information and I wanna cut them out. Of course the score too. But we will dicuss about it later at other repository.


https://drive.google.com/drive/folders/1z2GdAg8uz-ZCni1glbG1A-M6f7-R_6Y2?usp=sharing

Model

## 2. Dataset
* Word dataset for name: [ICFHR2018 Competition on Vietnamese Online Handwritten Text Recognition Database (HANDS-VNOnDB2018)](http://tc11.cvc.uab.es/datasets/HANDS-VNOnDB2018_1/) 
You can use `data/inkml_2_img.py` to covert ikml file into images
* Digit dataset for student ID and score: MNIST dataset. I have to generate multi - digit number from MNISt which you can find [here](https://github.com/mrzaizai2k/Multi-digit-images-generator-MNIST-) in my repo:

Those are raw file, of course they won't help at all without Data Augmentation
* Elastic Transform
* Adding blob, line noise
* Random Cutout
* Rotate and Scale

I applied them all in `source/prepare_MSSV_dataset.py` and `source/imgtocsv.py` for both name and student ID training. I found the those methods are not enough, so the solution is to collect more real data. I did add about 250 images for each, with Data augmentation I can make it to 20000 images and the result was good

## 3. Image Preprocessing
You can find code in `source/Preprocessing.py` 
The flow of this stage is:
1. Image Alignment 
2. Maximize Contrast
3. Otsu Threshold
4. Remove line/circle

When we first take the input image, we take the background information too, and the picture is not in the right direction which is hard to extract and recognize. With the help of Image Alignment, the work is much easier. 

<p align="center"><img src="doc/matches.jpg" width="500"></p>
<p align="center"><i>Hình 2. Image Alignment </i></p>

Reference: https://www.pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/

Then I crop Images I need with fixed pixels at all times

<p align="center"><img src="doc/MSSV_crop.jpg" width="200"></p>
<p align="center"><i>Hình 3. MSSV_crop.jpg </i></p>

I have to maximize contrast with [Top hat and Black hat method](https://www.quora.com/Why-use-the-top-hat-and-black-hat-morphological-operations-in-image-processing). I found this can hold back lots of information after Otsu Threshold, especially with blur image. I did compare between Adaptive Threshold and Otsu Theshold

Adaptive Threshold which we know that works really well with variations in lighting conditions, shadowing, etc... You can visit this [website](https://www.pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/) to know more. But it also retain noise. It's like **a lot of noise** which is hard to remove line and recognize even having Gaussian Blur step before. Otsu turns out performing so well, I guess that because the small image after croping reduce the affect of light variance.   

<p align="center"><img src="doc/removeline_122/namecrop_giaythi5.jpg" width="300"></p>
<p align="center"><i>Hình 4. Image after removing line </i></p>

## 4. Word segmentation
I have compare between EAST and scale Space techniques 

## 4. Imbalanced data

As you can see, our project is imbalanced positive: 669 (16.45% of total), negative cases: 3399 There are a alot of proposed methods to solve this like: 
* Over sampling
* Undersampling
* Hybrid over and under sampling
* Gain more data
* Data augmentation (I will use it for my next Covid classification phase 2)
   * Time stretch
   * Pitch shift
   * GAIN
   * Background noise
and so on...
     
Reference: https://phamdinhkhanh.github.io/2020/02/17/ImbalancedData.html#45-thu-th%E1%BA%ADp-th%C3%AAm-quan-s%C3%A1t

In this project. I'll try resolving the imbalanced data by oversampling with SMOTE. It's a oversampling method. For me the result is not really good because they change the 
features and we don't know how they change its and if the new features were true in real life. I guess in phase 2 I will try on Gain and background noise to oversample the dataset

However, SMOTE help a lot on the training time with early stopping

Reference: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

## 5. Model
I use a simple ANN model with drop out layers (0.5) to avoid overfitting. The model here is quite simple. I prefer using CRNN + attention and a more complicated model for
2D dataset instead of 1D dataset like this. You know, it's just **PHASE 1!**

## 6. K fold cross validation

Here I use K-fold (Stratified k fold) with the oversampled data

After all, I think K-fold is just a method to generally assessed how good or bad the model is. Help us tune the hyperparameters better

Reference:

https://viblo.asia/p/lam-chu-stacking-ensemble-learning-Az45b0A6ZxY

https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/

https://github.com/SadmanSakib93/Stratified-k-fold-cross-validation-Image-classification-keras/blob/master/stratified_K_fold_CV.ipynb

https://miai.vn/2021/01/18/k-fold-cross-validation-tuyet-chieu-train-khi-it-du-lieu/





