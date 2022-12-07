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
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data_sub/train_input/"
TRAIN_MASK_DIR = "data_sub/train_mask/"
VAL_IMG_DIR = "data_sub/val_input/"
VAL_MASK_DIR = "data_sub/val_mask/"
RES_DIR = "result/"
GROUPBY = 3
PAD_VALUE = 0
NUM_CLASS = 2
IGNORE_INDEX = -1
DISPLAY_STEP = 50
VAL_EVERY = 1
VAL_AFTER = 0



def checkpoint( log, config):
    with open(
        os.path.join(RES_DIR, "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]

def train_fn(loader,device, model, optimizer, loss_fn, scaler):
    #loop= tqdm(loader)
    
    """for batch_idx, (data, targets) in enumerate(loop):
        print("TENSOR SHAPE", data[0].shape)
        data = data[0].to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(input= data[0], batch_positions=data[1])
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())"""
    
    for i , batch in enumerate(loader):
        if device is not None:
            batch = recursive_todevice(batch,device)
        (x,dates), y = batch
        predictions = model(x,batch_positions = dates)

def iterate(
    model, data_loader, criterion, num_classes = 2, ignore_index = -1,display_step = 50,device_str = 'cuda', optimizer=None, mode="train", device=None
):
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
        y = y.long()

        if mode != "train":
            with torch.no_grad():
                out = model(x, batch_positions=dates)
        else:
            optimizer.zero_grad()
            out = model(x, batch_positions=dates)

        loss = criterion(out, y)
        if mode == "train":
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = out.argmax(dim=1)
        iou_meter.add(pred, y)
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
    """train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )"""
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
        BATCH_SIZE,
        #train_transform,
        #val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
        PAD_VALUE,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    
    #check_accuracy(val_loader, model, device=DEVICE)
    
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
        
        if epoch % VAL_EVERY == 0 and epoch >= VAL_AFTER:
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
            checkpoint(trainlog, RES_DIR)
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
                        RES_DIR, "model.pth.tar"
                    ),
                )
        else:
            trainlog[epoch] = {**train_metrics}
            checkpoint(trainlog, RES_DIR)
        print("Testing best epoch . . .")
        model.load_state_dict(
            torch.load(
                os.path.join(
                RES_DIR,  "model.pth.tar"
                )
            )["state_dict"]
        )
        
        """print(
            "Los {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
            val_metrics["test_loss"],
            val_metrics["test_accuracy"],
            val_metrics["test_IoU"],
            )
        )"""
            
        """model.eval()
        test_merics, conf_mat = iterate(
            modl,
            dat_loader=test_loader,
            crierion=criterion,
            conig=config,
            optmizer=optimizer,
            mod="test",
            devce=device,
        
        print
            "Los {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
            test_metrics["test_loss"],
            test_metrics["test_accuracy"],
            test_metrics["test_IoU"],
            
        
        save_reults(fold + 1, test_metrics, conf_mat.cpu().numpy(), config)"""
    

        # save model
        """checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)"""

        # check accuracy
        #check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        #print(DEVICE, val_loader.get_device(),model.get_device())
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()
