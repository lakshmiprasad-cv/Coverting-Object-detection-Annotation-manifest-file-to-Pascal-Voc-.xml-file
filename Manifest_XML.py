"""
__author__: Prasasd ALLU
Created: Saterday , 8th Jan 2021 3:49:37 pm 
"""
import os
from PIL import Image,ImageDraw 
import xml.etree.ElementTree as ET
import torch
from torchvision import transforms 
from torch.utils.data import Dataset
import numpy as np
import json
import random
from torchvision import transforms
import cv2
import os
import cv2
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
import numpy as np
from os.path import join
#outpath =
## converts the normalized positions  into integer positions
def unconvert(class_id, width, height, x, y, w, h):
    xmax = float(w)
    xmin = float(x)
    ymax = float(h)
    ymin = float(y)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)
def Manifest_xml(label_path='groundtruth/ground_truth/',data_path='groundtruth/ground_truth/images/',task = 'deck'):
    image_info = []
    classes = ['Background','deck', 'patio']
    with open(os.path.join(label_path,'output.manifest')) as f:
        lines = f.readlines()
        for line in lines:
            info = json.loads(line[:-1])
            if len(info[task]['annotations']):
                image_info.append(info)
    ids=[i['source-ref'].split('/')[-1].split('.')[0] for i in image_info]
    f=open('train.txt','w')
    for ele in ids:
        f.write(ele+'\n')
        

    for i in range(len(ids)):
        img_id = ids[i]
        info=image_info[i]
        image= cv2.imread((os.path.join(data_path,info['source-ref'].split('/')[-1])))
        height, width, channels = image.shape
        #print(height, width, channels)
        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'VOC2007'
        img_name = img_id + '.jpg'

        node_filename = SubElement(node_root, 'filename')
        node_filename.text = img_name

        node_source= SubElement(node_root, 'source')
        node_database = SubElement(node_source, 'database')
        node_database.text = 'The VOC2007 Database'

        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(width)

        node_height = SubElement(node_size, 'height')
        node_height.text = str(height)

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = str(channels)

        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = '0'
        boxess = info[task]['annotations']
        label_norm= boxess
        for i in range(len(label_norm)):
            labels_conv = label_norm[i]
            new_label = unconvert(labels_conv['class_id'], width, height, labels_conv['left'],labels_conv['top'], labels_conv['width']+labels_conv['left'], labels_conv['height']+labels_conv['top'])
           
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = classes[new_label[0]]
            print(new_label[0])
            node_pose = SubElement(node_object, 'pose')
            node_pose.text = 'Unspecified'


            node_truncated = SubElement(node_object, 'truncated')
            node_truncated.text = '0'
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(new_label[1])
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(new_label[3])
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text =  str(new_label[2])
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(new_label[4])
            xml = tostring(node_root, pretty_print=True)  
            dom = parseString(xml)
        root='VOC2'
        #annopath = join(root, 'labels', '%s.txt')
        #imgpath = join(root, 'images', '%s.jpg')
        os.makedirs(join(root, 'Annotations'), exist_ok=True)
        outpath = join(root, 'Annotations', '%s.xml')
        f =  open(outpath % img_id, "wb")
        f.write(xml)
        f.close()     


    
Manifest_xml(label_path='./',data_path='VOC2/JPEGImages',task = 'deck')
