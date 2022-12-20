import collections.abc
import re

import torch
import torchvision
from model.dataset import YeastDataset
import torch.nn as nn
import dill as pickle
from torch.utils.data import DataLoader
from torch.nn import functional as F




def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    groupby,
    mask_pos,
    batch_size,
    num_workers=4,
    pin_memory=True,
    pad_value = 0
):
    """" 
    load the training and test set
    Args:
        train_dir (string): training input directory
        train_maskdir (string): training mask directory
        val_dir (string): val input directory
        val_maskdir (string): val mask directory
        groupby (int): number of image taken by set
        batch size (int): batch
    Returns:
        train_loader: return element batch by batch
        test_loader : return element batch by batch
 
    """
    
    train_ds = YeastDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        mask_index = mask_pos,
        groupby = groupby,
    )
    

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        

    )

    val_ds = YeastDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        mask_index = mask_pos,
        groupby = groupby,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader




def save_predictions_as_imgs(
    loader, model, folder="image-save/", device="cuda"
):
    """" Predict and save the prediction and the true mask in a png file"""
    model.eval()
    for idx, ((x, dates), y) in enumerate(loader):
        x = x.to(device=device)
        dates = dates.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            out = model(x, batch_positions=dates).squeeze()
            pred = torch.round(torch.sigmoid(out)).float().cpu()

        
        
        if pred.dim() == 3: 
          torchvision.utils.save_image(
              pred.unsqueeze(1), f"{folder}/pred_{idx}.png"
          )
          torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
        else:
          torchvision.utils.save_image(
              pred, f"{folder}/pred_{idx}.png"
          )

          torchvision.utils.save_image(y, f"{folder}/{idx}.png")

        

    model.train()






