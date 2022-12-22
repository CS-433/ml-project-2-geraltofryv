from train_argparse import iterate, checkpoint, save_results
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from model.miou import IoU
from model.model import UTAE
import argparse 
import os
import json
import time
from model.metrics import confusion_matrix_analysis

from model.weight_init import weight_init
from model.utils import (
    get_loaders,
    save_predictions_as_imgs
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
PIN_MEMORY = True
LOAD_MODEL = False
PAD_VALUE = 0
NUM_CLASS = 2
IGNORE_INDEX = -1
DISPLAY_STEP = 50
VAL_EVERY = 1
VAL_AFTER = 0


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
      "--saved_images",
      default = "saved_images_TEST/",
      help= "folder where predicted images are stored"
    )
    parser.add_argument(
        "--res_dir",
        default="result/",
        help="Path to the folder where the results should be stored",
    )

    parser.add_argument(
      "--dataset",
      default="dtsub/",
      help = "Path containing dataset with train_input, train_mask, val_input and val_mask"
    )
    parser.add_argument(
      "--result",
      default="result/Groupby_5_result_augmented_100_epochs",
      help = "Path containing model.pth.tar"
    )
    parser.add_argument("--groupby", default=5, type=int, help="Number of time frames to consitute the temporal groups")
    parser.add_argument("--maskpos", default=2, type=int, help="Number of time frames to consitute the temporal groups")
    parser.add_argument(
      "--fold",
      default="Groupby_5_result_augmented_100_epochs/",
      help = "fold containing model.pth.tar"
    )
    parser.add_argument("--batch_size", default=3, type=int, help="Batch size")

    config=parser.parse_args()


    MODEL_PTH_SAVE = f"model_groupby_{config.groupby}_maskpos_{config.maskpos}.pth.tar"

    TRAIN_IMG_DIR = config.dataset +"train_input/"
    TRAIN_MASK_DIR = config.dataset +"train_mask/"
    VAL_IMG_DIR = config.dataset +"val_input/"
    VAL_MASK_DIR = config.dataset +"val_mask/"

    SAVED_IM = config.saved_images
    RES_DIR = config.res_dir
    GROUPBY= config.groupby
    MASK_POS = config.maskpos
    FOLD_GROUPBY = config.fold
    BATCH_SIZE = config.batch_size
    

    if not os.path.exists(SAVED_IM):
      # Create a new directory because it does not exist
      os.makedirs(SAVED_IM)
      print("The new directory is created! ", SAVED_IM)

    device = torch.device(DEVICE)
    model = UTAE(input_dim=3).to(device)
    model.apply(weight_init)
    model.load_state_dict(
            torch.load(
                os.path.join(
                RES_DIR,FOLD_GROUPBY,  MODEL_PTH_SAVE
                )
            )["state_dict"]
        )
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        GROUPBY,
        MASK_POS,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        PAD_VALUE,
    )

    save_predictions_as_imgs(
            val_loader, model, folder=SAVED_IM, device=DEVICE
        )


if __name__ == "__main__":
    main()
