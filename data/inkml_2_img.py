import os
import xml.etree.ElementTree as ET
import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
import pyprind
import warnings
warnings.filterwarnings('ignore')

dataFolder = "./InkData_word"
targetFolder = dataFolder + "_processed"
lineWidth = 10

if not os.path.exists(targetFolder): 
    os.mkdir(targetFolder)

def get_traces_data(file_path):
    ''' get traces from .inkml file format '''
    tree = ET.parse(file_path)
    root = tree.getroot()

    trace_all = []
    labels = []
    # root to childs include 'annotationXML' and 'traceGroup's tag
    for child in root:
        # ignore if tag is 'annotationXML'
        if child.tag == 'annotationXML':
            continue
        trace_group = []
        # childs to 'annotationXML' and 'trace's tag
        for trace in child:
            # get ground truth 
            if trace.tag == 'annotationXML':
                labels.append({'id': child.get('id'), 'ground_truth': trace[0].text})
                continue

            trace_line = []
            text = (trace.text).replace('\n', '').split(',')
            # get trace line
            for ch in text:
                temp = ch.split(' ')
                trace_line.append([round(float(temp[0])), round(float(temp[1]))])
            # trace in 'traceGroups'       
            trace_group.append({'coords': trace_line})
        # get all traces
        trace_all.append({'id': child.get('id'), 'trace': trace_group})

    return trace_all, labels

def makeLabelingImage(input_path, output_base):
    ''' make and save labels and images convert from .inkml file format '''
    traces, ground_truth = get_traces_data(input_path)
    output_base_labels = targetFolder + '/' + output_base + '/labels/'
    output_base_images = targetFolder + '/' + output_base + '/images/'

    if not os.path.exists(targetFolder + '/' + output_base):
        os.mkdir(targetFolder + '/' + output_base)
    if not os.path.exists(output_base_labels): 
        os.mkdir(output_base_labels)
    if not os.path.exists(output_base_images): 
        os.mkdir(output_base_images)
    
    pbarGT = pyprind.ProgBar(len(ground_truth), title='Lables')
    # save labels from ground truth values
    for line in ground_truth:
        with open(output_base_labels + str(line['id']) + '.txt', 'w', encoding='utf-8') as label:
            label.write(str(line['ground_truth']))
            pbarGT.update() 
    # convert and save images
    pbarTr = pyprind.ProgBar(len(traces), title='Images')
    for trace_group in traces:
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.axes().spines['top'].set_visible(False)
        plt.axes().spines['right'].set_visible(False)
        plt.axes().spines['bottom'].set_visible(False)
        plt.axes().spines['left'].set_visible(False)

        for trace_line in trace_group['trace']:
            ls = trace_line['coords']
            data = np.array(ls)
            x, y = zip(*data)
            plt.plot(x, y, linewidth=lineWidth, c='black')

        plt.savefig(output_base_images + trace_group['id'] + '.png', bbox_inches='tight', dpi=100)
        plt.gcf().clear()
        pbarTr.update()


if __name__ == '__main__':
    counts = 1
    for file in os.listdir(dataFolder):
        fileName = file.split('.')[0]
        print('\n---------Processing: %d/255---------' % counts)
        makeLabelingImage(dataFolder+'/'+file, fileName)
        counts += 1