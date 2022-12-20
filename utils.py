import collections.abc
import re

import torch
import torchvision
from dataset import YeastDataset
import torch.nn as nn
import dill as pickle
from torch.utils.data import DataLoader
from torch.nn import functional as F

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


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


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder="image-save/", device="cuda"
):
    model.eval()
    for idx, ((x, dates), y) in enumerate(loader):
        x = x.to(device=device)
        dates = dates.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            out = model(x, batch_positions=dates).squeeze()
            pred = torch.round(torch.sigmoid(out)).float().cpu()
            #pred = torch.squeeze(pred)

        
        #y = y.squeeze().float().cpu()
        print("PRED SHAPE", pred.shape)
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






