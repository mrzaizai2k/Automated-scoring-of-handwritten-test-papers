import pandas as pd
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from create_metrics_OCR import _levenshtein_distance

def class_list(dataPath):

    data = pd.read_excel(dataPath)

    MSSV_list = data.iloc[: ,0].to_list()
    Ho_list = data.iloc[: ,1].to_list()
    Ten_list = data.iloc[: ,2].to_list()
    name_MSSV_list = []
    name_list = []
    for i in range (len(Ho_list)):
        name_list.append(str(Ho_list[i]+' '+Ten_list[i])) 
        name_MSSV_list.append(str(Ho_list[i]+' '+Ten_list[i] + ' ' + str(MSSV_list[i])))
    return name_list, MSSV_list, name_MSSV_list

def lexicon_search(hypo, dic):

    dis_list =[]
    index = 0
    for i in range (len(dic)):
        dic[i] = str(dic[i])
        distance  = _levenshtein_distance(hypo.lower(),dic[i].lower())
        dis_list.append(distance)
        if distance == min (dis_list):
            index = i 
    return index, dic[index]


def main():
    print ('distance',_levenshtein_distance('Mai chí Bảo','Mai Chí Bảo'))
    class_list_dir = 'data\Class_list.xlsx'
    name_list, MSSV_list = class_list(class_list_dir)
    name_index, name_recognized = lexicon_search ("NMai Khí bbảo", name_list)
    print('\nNAME_Recognized: ' + name_recognized+'\n----------')

if __name__ == '__main__':
    main()


