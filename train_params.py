from train import iterate, checkpoint, save_results
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from miou import IoU
from model import UTAE
import os
import json
import time
from metrics import confusion_matrix_analysis

from weight_init import weight_init
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

LEARNING_RATE = 1e-4
#DEVICE = "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 3
NUM_EPOCHS = 2
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "dtsub/train_input/"
TRAIN_MASK_DIR = "dtsub/train_mask/"
VAL_IMG_DIR = "dtsub/val_input/"
VAL_MASK_DIR = "dtsub/val_mask/"
RES_DIR = "result/"
GROUPBY_list = [3,5,7,9,11]
PAD_VALUE = 0
NUM_CLASS = 2
IGNORE_INDEX = -1
DISPLAY_STEP = 50
VAL_EVERY = 1
VAL_AFTER = 0
ALL_MASK_list = [list(range(gp)) for gp in GROUPBY_list]
MODEL_PTH_SAVE_list = []
FOLD_GROUPBY_list = []
for i, gp in enumerate(GROUPBY_list):
    tmp = []
    FOLD_GROUPBY_list.append(f'Groupby_{gp}_result')
    for ms in ALL_MASK_list[i]: 
        tmp.append(f"model_groupby_{gp}_maskpos_{ms}.pth.tar")
    
    MODEL_PTH_SAVE_list.append(tmp)



for fold in FOLD_GROUPBY_list:

    # Check whether the specified path exists or not
    directory = os.path.join(RES_DIR,fold)
    isExist = os.path.exists(directory)
    if not isExist:

   # Create a new directory because it does not exist
        os.makedirs(directory)
        print("The new directory is created!")





def main():

    for p, GROUPBY in enumerate(GROUPBY_list):
      FOLD_GROUPBY = FOLD_GROUPBY_list[p]
      print("Groupby : ", GROUPBY, FOLD_GROUPBY)

      for m, MASK_POS in enumerate(ALL_MASK_list[p]):
          MODEL_PTH_SAVE = MODEL_PTH_SAVE_list[p][m]
          print("Mask pos: ", MASK_POS, MODEL_PTH_SAVE )
              
          device = torch.device(DEVICE)

          print(device)
          model = UTAE(input_dim=3).to(device)
          model.apply(weight_init)
          weights = torch.ones(NUM_CLASS, device=DEVICE).float()
          weights[IGNORE_INDEX] = 0
          criterion = nn.CrossEntropyLoss(weight=weights)
          loss_fn = nn.BCEWithLogitsLoss()
          optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

          """if LOAD_MODEL:
              load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)"""

          
          
          
          trainlog = {}
          best_mIoU = 0
          for epoch in range(NUM_EPOCHS):
              print("EPOCH number ->", epoch)
              #train_fn(train_loader, DEVICE, model, optimizer, loss_fn, scaler)
              print("EPOCH {}/{}".format(epoch, NUM_EPOCHS))

              model.train()
              train_metrics = iterate(
                  model,
                  data_loader=train_loader,
                  criterion=criterion,
                  num_classes = NUM_CLASS, 
                  ignore_index = IGNORE_INDEX, 
                  display_step = DISPLAY_STEP,
                  device_str = DEVICE,
                  optimizer=optimizer,
                  mode="train",
                  device=device,
              )
              print("Train set passed iterate function")
              
              if epoch % VAL_EVERY == 0:
                  print("Validation . . . ")
                  model.eval()
                  val_metrics = iterate(
                      model,
                      data_loader=val_loader,
                      criterion=criterion,
                      num_classes = NUM_CLASS, 
                      ignore_index = IGNORE_INDEX, 
                      display_step = DISPLAY_STEP,
                      device_str = DEVICE,
                      optimizer=optimizer,
                      mode="val",
                      device=device,
                  )
                  print(
                      "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
                          val_metrics["val_loss"],
                          val_metrics["val_accuracy"],
                          val_metrics["val_IoU"],
                      )
                  )
                  trainlog[epoch] = {**train_metrics, **val_metrics}
                  checkpoint(trainlog, FOLD_GROUPBY, RES_DIR)
                  print('val metrics iou HERE', val_metrics["val_IoU"])
                  if val_metrics["val_IoU"] >= best_mIoU:
                      best_mIoU = val_metrics["val_IoU"]
                      print("TORCH.SAVE HERE")
                      torch.save(
                          {
                              "epoch": epoch,
                              "state_dict": model.state_dict(),
                              "optimizer": optimizer.state_dict(),
                          },
                          os.path.join(
                              RES_DIR,FOLD_GROUPBY, MODEL_PTH_SAVE
                          ),
                      )
              else:
                  trainlog[epoch] = {**train_metrics}
                  checkpoint(trainlog, FOLD_GROUPBY, RES_DIR)
              print("Testing best epoch . . .")
              model.load_state_dict(
                  torch.load(
                      os.path.join(
                      RES_DIR,FOLD_GROUPBY,  MODEL_PTH_SAVE
                      )
                  )["state_dict"]
              )
              
              

              
          save_results(trainlog,FOLD_GROUPBY, RES_DIR, GROUPBY, MASK_POS )




if __name__ == "__main__":
    main()