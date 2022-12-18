import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import more_itertools as mit
import torch
import re
import cv2 as cv


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class YeastDataset(Dataset):
    def __init__(self, image_dir, mask_dir, mask_index, groupby = 3):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        images = os.listdir(image_dir)
        images.sort(key= natural_keys)
        num_of_set = []
        group_set = []
        id_image = []
        for i, t in enumerate(images):
            num_of_set.append(int(re.search('set_(.*)_time', t)[1]))
            
        
        for r in list(set(num_of_set)):
            tmp = []
            tmp_id = []
            j = 0
            for i,t in enumerate(images):
                
                if t.find(f"set_{r}") != -1:
                    tmp_id.append(j)
                    tmp.append(t)
                    j += 1
            id_image.append(tmp_id)
            group_set.append(tmp)
        self.group_set = group_set
        groupby_each_set = []
        
        group_id = []
        for i, val in enumerate(list(set(num_of_set))):    
            groupby_each_set.append(list(mit.windowed(group_set[i], n=groupby, step=1)))
            group_id.append(list(mit.windowed(id_image[i], n=groupby, step=1)))
            
        self.group_consecutive_time = [item for sublist in groupby_each_set for item in sublist]
        self.group_id = [item for sublist in group_id for item in sublist]

        self.group_set = group_set
        
        self.mask_index = mask_index


    def __len__(self):
        return len(self.group_consecutive_time)

    def __getitem__(self, index):
        
        set_images = []
        #set_masks = []
        index_id = self.group_id[index]
        for i, name in enumerate(self.group_consecutive_time[index]):
            img_path = os.path.join(self.image_dir, name)
            img = np.array(Image.open(img_path).convert("RGB"))
            set_images.append(np.array(Image.open(img_path).convert("RGB")))
            

        set_images = np.array(set_images)
        #middleIndex = int((len(self.group_consecutive_time[index]) - 1)/2) # pick the middle mask
        mask_path = os.path.join(self.mask_dir, self.group_consecutive_time[index][self.mask_index]).replace("_input.png", "_mask.png")
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        _,mask = cv.threshold(mask,127, 1, cv.THRESH_BINARY)
        #print(mask)
        #mask[mask > 30] = 1.0
        #mask[mask == 30] = 0.0
        set_images = np.moveaxis(set_images, 3,1)
        set_im_torch = torch.from_numpy(set_images).float()
        norm = torch.nn.InstanceNorm2d(3)
        set_norm_img = norm(set_im_torch)
        mask_torch = torch.from_numpy(mask).float()
        #mask_torch_GS = mask_torch[None,:]
        index_id = torch.from_numpy(np.array(index_id))
        

        return (set_norm_img, index_id ), mask_torch
    








        






