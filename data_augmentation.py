import os
from PIL import Image
from anyio import getnameinfo
from torch.utils.data import Dataset
import numpy as np
import more_itertools as mit
import torch
import re
import torchvision

import tensorflow as tf
from keras import layers

TRAIN_IMG_DIR = "data_sub/train_input/"
TRAIN_MASK_DIR = "data_sub/train_mask/"

def rot_and_flip(img) :
    processed_img = tf.image.rot90(img)
    processed_img = tf.image.flip_up_down(img)
    return(processed_img.numpy())


def adjust_sat_and_bright(img) :
    processed_img = tf.image.adjust_contrast(img, 0.3)
    processed_img = tf.image.adjust_gamma(processed_img,0.5)
    return(processed_img.numpy())

def crop(img) :
    processed_img = tf.image.central_crop(img,0.5)
    processed_img = tf.image.resize(processed_img,[256,256])
    return(processed_img.numpy())

last_file = os.listdir(TRAIN_IMG_DIR)[-1]
og_set_nb = re.findall(r'\d+', last_file)[0]

both_dirs = [TRAIN_IMG_DIR,TRAIN_MASK_DIR]

for dir in both_dirs :

    end_file_name = ""
    convert_type = "RGBA"

    if dir == TRAIN_IMG_DIR :
        end_file_name = "_input.png"
    else :
        end_file_name = "_mask.png"

    for entry in os.scandir(dir):
        searching_set_0 = "set_"+str(0)+"_time_" 
        searching_set_1 = "set_"+str(1)+"_time_"  
        searching_set_2 = "set_"+str(2)+"_time_"   

        name  = entry.name

        if searching_set_0 in name:
            set_nb, timepoint = re.findall(r'\d+', name)[0],re.findall(r'\d+', name)[1]
            set_0_img = np.array(Image.open(dir + name).convert(convert_type))
            rotated_img = rot_and_flip(set_0_img)
            tf.keras.utils.save_img(dir + "set_"+str(int(og_set_nb)+1)+"_time_"+str(timepoint)+end_file_name,rotated_img, data_format = "channels_last")
            #torchvision.utils.save_image(rotated_img, dir + "set_"+str(int(og_set_nb)+1)+"_time_"+str(timepoint)+end_file_name)

        if searching_set_1 in name :
            set_nb, timepoint = re.findall(r'\d+', name)[0],re.findall(r'\d+', name)[1]
            set_1_img = np.array(Image.open(dir + name).convert(convert_type), dtype=np.float32)
            adj_img = adjust_sat_and_bright(set_1_img)
            tf.keras.utils.save_img(dir + "set_"+str(int(og_set_nb)+2)+"_time_"+str(timepoint)+end_file_name,adj_img,data_format = "channels_last")

        if searching_set_2 in name :
            set_nb, timepoint = re.findall(r'\d+', name)[0],re.findall(r'\d+', name)[1]
            set_2_img = np.array(Image.open(dir + name).convert(convert_type), dtype=np.float32)
            flipped_img = crop(set_2_img) 
            tf.keras.utils.save_img(dir + "set_"+str(int(og_set_nb)+3)+"_time_"+str(timepoint)+end_file_name,flipped_img,data_format = "channels_last")

print("New training sets and their corresponding masks have been generated in data_sub/train_input/ and data_sub/train_mask/")
