import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from model.miou import IoU
from model.model import UTAE
import os
import json
import time
from model.metrics import confusion_matrix_analysis
import argparse 
from csv import writer
from numpy import asarray
from numpy import savetxt
import shutil

from model.weight_init import weight_init
from model.utils import (
    get_loaders,
    save_predictions_as_imgs
)


#Initialization of data

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



def checkpoint( log,fold_groupby, res_dir):
    """"
    Save the result of one epoch in a json file
    Args:
        - log: result of the epoch in list
        - fold_groupby (String): folder where to store json file
        - res_dir (String): folder directory
    """
    with open(
        os.path.join(res_dir, fold_groupby, "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)

def recursive_todevice(x, device):
    """"
    taken from the article "Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks"
        """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]

def save_results(metrics, fold_groupby, res_dir, groupby, mask_pos ):
    with open(
        os.path.join(res_dir,fold_groupby, f"groupby_{groupby}_maskpos_{mask_pos}_metric.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)

def iterate(
    model, data_loader, criterion, num_classes = 2, ignore_index = -1,display_step = 50,device_str ='cuda', optimizer=None, mode="train", device=None
):
    """"
    taken from the article "Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks"
    Train the model and return metrics
    """
    loss_meter = tnt.meter.AverageValueMeter()
    iou_meter = IoU(
        num_classes=num_classes,
        ignore_index=ignore_index,
        cm_device=device_str,
    )

    t_start = time.time()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), y = batch
        y = y.long().squeeze(1)

        if mode != "train":
            with torch.no_grad():
                out = model(x, batch_positions=dates).squeeze(1)
                pred = torch.round(torch.sigmoid(out))
        else:
            optimizer.zero_grad()
            out = model(x, batch_positions=dates).squeeze(1)
            with torch.no_grad():
              pred = torch.round(torch.sigmoid(out))

        loss = criterion(out.squeeze(), y.float().squeeze(0).squeeze(1))
        if mode == "train":
            loss.backward()
            optimizer.step()

        iou_meter.add(pred.long(), y)
        loss_meter.add(loss.item())

        if (i + 1) % display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            print(
                "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, mIoU {:.2f}".format(
                    i + 1, len(data_loader), loss_meter.value()[0], acc, miou
                )
            )

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    miou, acc = iou_meter.get_miou_acc()
    metrics = {
        "{}_accuracy".format(mode): acc,
        "{}_loss".format(mode): loss_meter.value()[0],
        "{}_IoU".format(mode): miou,
        "{}_epoch_time".format(mode): total_time,
    }

    if mode == "test":
        return metrics, iou_meter.conf_metric.value()  # confusion matrix
    else:
        return metrics


def main():

    """"
    We use a parser to tune different parameter the location of the directories, the number of epochs, batch size,
    learning rate, groupby and the position of the mask
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--res_dir",
        default="results/",
        help="Path to the folder where the results should be stored",
    )
    parser.add_argument(
      "--saved_images",
      default = "saved_images/",
      help= "folder where predicted images are stored"
    )
    parser.add_argument(
      "--dataset",
      default="dtsub/",
      help = "Path containing dataset with train_input, train_mask, val_input and val_mask"
    )
    parser.add_argument("--epochs", default=2, type=int, help="Number of epochs per fold")
    parser.add_argument("--batch_size", default=3, type=int, help="Batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--groupby", default=3, type=int, help="Number of time frames to consitute the temporal groups")
    parser.add_argument("--maskpos", default=0, type=int, help="Number of time frames to consitute the temporal groups")
    config=parser.parse_args()

    # directory of the dataset
    TRAIN_IMG_DIR = config.dataset +"train_input/"
    TRAIN_MASK_DIR = config.dataset +"train_mask/"
    VAL_IMG_DIR = config.dataset +"val_input/"
    VAL_MASK_DIR = config.dataset +"val_mask/"

    # directory where saving the different results
    GROUPBY= config.groupby
    MASK_POS = config.maskpos
    FOLD_GROUPBY = f'Groupby_{GROUPBY}_result'
    MODEL_PTH_SAVE = f"model_groupby_{GROUPBY}_maskpos_{MASK_POS}.pth.tar"
    SAVED_IM = config.saved_images
    RES_DIR = config.res_dir
    
    # Parameters concerning the tuning of the model
    BATCH_SIZE = config.batch_size
    NUM_EPOCHS = config.epochs
    LEARNING_RATE = config.lr
    
    
    if not os.path.exists(RES_DIR):
      # Create a new directory because it does not exist
      os.makedirs(RES_DIR)
      print("The new directory is created! ", RES_DIR) 
    

    if not os.path.exists(SAVED_IM):
      # Create a new directory because it does not exist
      os.makedirs(SAVED_IM)
      print("The new directory is created! ", SAVED_IM)

    if not os.path.exists(RES_DIR):
      # Create a new directory because it does not exist
      os.makedirs(RES_DIR)
      print("The new directory is created!")

    directory = os.path.join(RES_DIR,FOLD_GROUPBY)
    if not os.path.exists(directory):
      # Create a new directory because it does not exist
      os.makedirs(directory)
      print("The new directory is created!")
    
    device = torch.device(DEVICE)
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

    

    
    #check_accuracy(val_loader, model, device=DEVICE)
    
    trainlog = {}
    best_mIoU = 0
    train_metrics_arr=[]
    val_metrics_arr=[]
    for epoch in range(NUM_EPOCHS):
        print("EPOCH number ->", epoch)
        print("EPOCH {}/{}".format(epoch+1, NUM_EPOCHS))

        model.train()
        train_metrics = iterate(
            model,
            data_loader=train_loader,
            criterion=loss_fn,
            num_classes = NUM_CLASS, 
            ignore_index = IGNORE_INDEX, 
            display_step = DISPLAY_STEP,
            device_str = DEVICE,
            optimizer=optimizer,
            mode="train",
            device=device,
        )
        train_metrics_arr.append(train_metrics)
        print("Train set passed iterate function")
        
        if epoch % VAL_EVERY == 0 and epoch >= VAL_AFTER:
            print("Validation . . . ")
            model.eval()
            val_metrics = iterate(
                model,
                data_loader=val_loader,
                criterion=loss_fn,
                num_classes = NUM_CLASS, 
                ignore_index = IGNORE_INDEX, 
                display_step = DISPLAY_STEP,
                device_str = DEVICE,
                optimizer=optimizer,
                mode="val",
                device=device,
            )
            val_metrics_arr.append(val_metrics)
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

        save_predictions_as_imgs(
            val_loader, model, folder=SAVED_IM, device=DEVICE
        )
    
        
    save_results(trainlog,FOLD_GROUPBY, RES_DIR, GROUPBY, MASK_POS )
    print("BEST mIoU:", best_mIoU)
        


if __name__ == "__main__":
    main()
