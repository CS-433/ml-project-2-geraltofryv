# Machine Learning Project 2 - 2022 - Team Geralt of Ryv

## The project:
In this repository, we implement U-Net with Temporal Attention Encoder (UTAE) model to evaluate the efficiency of temporal attention for budding yeast cells segmentation.
Through a binary classification, the task is to predict a mask that segment cells.
Initially, the data are stored in tiff file in the folder "cropped". To get access to our final dataset, you have to run image_preprocessing/split_data_crop_set_3.py, which extract the set of images and split them on train and test set and store them in "dtsub" folder, and image_preprocessing/data_augmentation.py, which creates more data on the training set.
The external libraries that we use were:
    - torch and torchnet to implement deep learning model
    - tensorflow to get augmented data


## Team:
  - Romain Rochepeau : romain.rochepeau@epfl.ch
  - Yassine Jamaa : yassine.jamaa@epfl.ch
  - Virginie Garnier : virginie.garnier@epfl.ch 

## Datasets:
 In order to train and test our different models, we used the following datasets: 
 ### dtsub/train_input and dtsub/train_mask - Training set of 807 images:
 - 256 x 256 image size
 - The training set is constituted of the fourth first set of images + the transformation
 - Input images are in RGB
 - Mask images are in binary. 0 if the pixel belong to background and 1 if the pixel belong to a budding yeast cells.


 ### dtsub/val_input and dtsub/val_mask - Training set of 121 images:
 - Everything as above, except the transformation


##  Architecture
To organize our code, we divide the used functions into different Python files:

- Folder `image_preprocessing`: contain `split_data_crop_set_3.py` and `data_augmentation.py` which allow to generate our wanted datas from `cropped` folder.
- Folder `model`: contains the python files used for the construction of UTAE model, the creation of the dataset and the calculation of metrics
        - `dataset.py`: loading the dataset
        - `metrics.py`: calculate the confusion matrix and mIoU(code from the article "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal    Attention Network")
        - `mIoU.py`: calculate IoU accuracy (code from the article "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal Attention Network")
        - `model.py`: implementation of UTAE model (code from the article "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal Attention Network")
        - `mltae.py`: implementation of Lightweight Temporal Attention Encoder (L-TAE) for image time series and Multi-Head Attention module (code from the article "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal Attention Network")
        - `positional_encoding.py`: implementation of positional encoder (code from the article "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal Attention Network")
        - `weight_init.py`: Initializes a model's parameters (code from the article "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal Attention Network")
        - `utils.py`: load the training and test set and save the predicted images.

- `train_params.py`: hyperparameter tuning of our UTAE model and store the results in folder `result_augmented`
- `train_argparse.py`: train UTAE model with specific parameters.
- Folder `plot_graph_result`: contain `plot_results.ipynb` which is used too plot the results that we get from `train_params.py`
    
## UTAE Model


