import tifffile
import matplotlib.pyplot as plt
import os
import re
import shutil
import numpy as np
import torch
import torch.utils.data as data
import torchnet as tnt
import time
import cv2
import pandas as pd
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.image

#Folder path
#initialization data
folder_data = "../ds" # where we store the dataset
split_train_test = "../dtsub" #folder where we divided the dataset into train_input train_mask val_input val_mask

# Function to sort the string filename corresponding to the set and time
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# Take .tif file and convert them to .png file and store them to a folder

directory = os.getcwd()
goal_dir = os.path.join(os.getcwd(), "../cropped")
PATH_DATA = os.path.abspath(goal_dir)
content_name = os.listdir(PATH_DATA)
input_data_name = [t for t in content_name if t.endswith('im.tif')]
output_mask_name = [t for t in content_name if t.endswith('mask.tif')]

data = [tifffile.TiffFile(os.path.abspath(os.path.join(PATH_DATA, t))) for t in input_data_name]
mask = [tifffile.TiffFile(os.path.abspath(os.path.join(PATH_DATA, t))) for t in output_mask_name]


# create the folder 'dataset' where all the new images is gonna be store


# Check whether the specified path exists or not
path_dataset = os.path.join(os.getcwd(), folder_data)
isExist = os.path.exists(path_dataset)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path_dataset)
   print("The new directory is created!")

cat = ['input', 'mask']
for i , fold in enumerate(cat):
    path_images = os.path.join(path_dataset, fold)
    isExist = os.path.exists(path_images)
    if not isExist:
        os.makedirs(path_images)
        print("The new directory is created!")


# save the image in png format in a folder

for i, file in enumerate(data):
    for j in range(len(file.pages)):
        path_input = os.path.join(path_dataset, cat[0])
        path_mask = os.path.join(path_dataset, cat[1])
        path_image_input = f'set_{i}_time_{j}_{cat[0]}.png'
        path_image_mask = f'set_{i}_time_{j}_{cat[1]}.png'
        #print(file.pages[j].asarray().astype(np.float32).dtype)
        mask_arr = mask[i].pages[j].asarray().astype(np.float32)
        mask_arr[mask_arr != 0] = 1
        
        matplotlib.image.imsave(os.path.join(path_mask, path_image_mask), mask_arr)
        matplotlib.image.imsave(os.path.join(path_input, path_image_input), file.pages[j].asarray().astype(np.float32))

print("Extraction of png file")

# set 3 have a different format size and contain two budding yeast colonies. We crop this image in two

image_directory = folder_data +'/input'
mask_directory = folder_data +'/mask'
list_input = os.listdir(image_directory)

set_3_list = []
for i,t in enumerate(list_input):
    if t.find(f"set_{3}") != -1:
         set_3_list.append(t)

set_3_list.sort(key=natural_keys)

for i, element in enumerate(set_3_list):
    path_im = os.path.join(image_directory, element)
    path_mask = os.path.join(mask_directory, element).replace("_input.png", "_mask.png")
    im = Image.open(path_im)
    mask = Image.open(path_mask)
    width, height = im.size
    #crop from the right side of the image
    left_1 = width - 256
    top_1 = 0
    right_1 = width
    bottom_1 = 256

    im_R = im.crop((left_1, top_1, right_1, bottom_1))
    mask_R = mask.crop((left_1, top_1, right_1, bottom_1))
    
    path_im_R = os.path.join(image_directory, f"set_3_time_{i}_input.png")
    path_mask_R = os.path.join(mask_directory, f"set_3_time_{i}_mask.png")
    imR = im_R.save(path_im_R)
    maskR = mask_R.save(path_mask_R)

    #crop from the left size of the image
    left_2 = 0
    top_2 = 0
    right_2 = 256
    bottom_2 = 256
    im_L = im.crop((left_2, top_2, right_2, bottom_2))
    mask_L = mask.crop((left_2, top_2, right_2, bottom_2))
    path_im_L = os.path.join(image_directory, f"set_4_time_{i}_input.png")
    path_mask_L = os.path.join(mask_directory, f"set_4_time_{i}_mask.png")
    imL = im_L.save(path_im_L)
    maskL = mask_L.save(path_mask_L)




# delete empty image in set_3 that are located in time 0 to time 39

empty_image = set_3_list[:40]
for i, el in enumerate(empty_image):
    input_del = os.path.join(image_directory, el)
    mask_el = el.replace("_input.png", "_mask.png")
    mask_del = os.path.join(mask_directory, mask_el)
    os.remove(input_del)
    os.remove(mask_del)

print("Cropping the set 3 into two set: set 3 and set 4")


# We create a new folder where we're gonna split our dataset in two part: training set and test set.
# We take the set 0,1,2 and 3 as training set and we take set 4 for the test set

# Check whether the specified path exists or not
path_dataset = os.path.join(os.getcwd(), split_train_test)
isExist = os.path.exists(path_dataset)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path_dataset)
   print("The new directory is created!")

cat = ["train_input", "train_mask", "val_input", "val_mask"]
for i , fold in enumerate(cat):
    path_images = os.path.join(path_dataset, fold)
    isExist = os.path.exists(path_images)
    if not isExist:
        os.makedirs(path_images)
        print("The new directory is created!")



list_input = os.listdir(image_directory)
list_input.sort(key=natural_keys)
set_4_list = []
for i,t in enumerate(list_input):
    if t.find(f"set_{4}") != -1:
         set_4_list.append(t)

list_input_without_set_4 = [x for x in list_input if x not in set_4_list]

# we copy paste the png file in the corresponding directory
for i, el in enumerate(list_input_without_set_4):
    input_original = os.path.join(image_directory, el)
    mask_el = el.replace("_input.png", "_mask.png")
    mask_original = os.path.join(mask_directory, mask_el)
    target_input = os.path.join(path_dataset, "train_input", el)
    target_mask = os.path.join(path_dataset, "train_mask", mask_el)
    shutil.copyfile(input_original, target_input)
    shutil.copyfile(mask_original, target_mask)


for i, el in enumerate(set_4_list):
    input_original = os.path.join(image_directory, el)
    mask_el = el.replace("_input.png", "_mask.png")
    mask_original = os.path.join(mask_directory, mask_el)
    target_input = os.path.join(path_dataset, "val_input", el)
    target_mask = os.path.join(path_dataset, "val_mask", mask_el)
    shutil.copyfile(input_original, target_input)
    shutil.copyfile(mask_original, target_mask)

print("Dataset divided into training and test set. Done.")