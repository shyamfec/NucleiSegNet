# NucleiSegnet-Paper-with-Code
This repository contains the Tensorflow code for the paper:
[NucleiSegNet: Robust deep learning architecture for the nuclei segmentation of liver cancer histopathology images](https://www.sciencedirect.com/science/article/abs/pii/S0010482520304066)

If you find this code helpful, please cite our work:
```
@article{LAL2021104075,
title = {NucleiSegNet: Robust deep learning architecture for the nuclei segmentation of liver cancer histopathology images},
journal = {Computers in Biology and Medicine},
volume = {128},
pages = {104075},
year = {2021},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2020.104075},
url = {https://www.sciencedirect.com/science/article/pii/S0010482520304066},
author = {Shyam Lal and Devikalyan Das and Kumar Alabhya and Anirudh Kanfade and Aman Kumar and Jyoti Kini},
keywords = {Deep neural network, Histopathology image, Nuclei segmentation, Nuclei detection},
}
```
## Goal
To perform nuclei segmentation of the Liver cancer Histopathology Images using deep learning architecture.

## Requirements
All the scripts ran using Tensorflow v2.2 in Google Colab Notebook. So all the required dependencies can be installed using
```pip install -r requirements.txt```

## Usage
* It is necessary to run the ```pip install spams``` as the stain normalization requires this package.Then you can run the other scripts
* Perform the stain normalization and cropping into patches of 16 (Multi-organ dataset has image size of 1000 x 1000). This will be done by running ``` python color_norm.py```. You can give here the image sizes and path to folders (train,test,valid) and also path where the cropped normalized patches can be saved.
* Then run the ```python main.py``` which will start the training process. It requires path to the saved color normalized patches and various other parametrs. You can change all at the top of the scripts
* For the test, run the ```python test.py``` which will carry out the inference on the test set. But remeber to give the path to your saved weights, model and test image patches.
* For overlaying the mask on the test image, run ```python overlay_mask.py``` . Remember to give the correct path at the start of the script 
* All the scripts, are run together in Google colab, to show how to run the scripts.

## Script Dependencies (code summary)
```
code-template/
│
├── color_norm.py           - script to stain normalize the images and crop into patches.
├── main.py                 - Script to run the dataset generator and start training 
│   ├── dataset.py          - dataset class and the augmentations used
│   ├── model.py            - script of the model
│       ├── loss_metric.py  - The metrics and loss functions for the model
│   
├── test.py                 - the script to test the model 
│   ├── dataset.py          - dataset class and the augmentations used
│   ├── model.py            - script of the model
│   ├── scores_comp.py      - computes the scores (DICE, IOU)
│   ├── color_norm.py       - joins the cropped patches back into full sized image
│ 
├── overlay_mask.py         - the script to overlay the segmented output on the test image
```

## Folder Structure while running the code
The code is built keeping this folder template in mind. So by default this folder sturcture will be created automatically. You need to specify the paths for these folders while running the code.  You can change it but maintain the 'Bin' and 'tis' part for labels and images.
```
Folder Template/
│
└── TrainData/     - Training data folder containing the stain normalized patches for training
    ├── Bin        - Binary Labels
    ├── tis        - Training images
└── ValidData/     
    ├── Bin  
    ├── tis 
└── TestData/      
    ├── Bin  
    ├── tis 
├── checkpoint     - For saving the weight
├── history        - For saving the loss and metrice score
├── results        - For saving the segmented mask
├── results        - For saving the segmented mask
├── overlay        - For saving the overlaid final result
```
## Model weght
NucleiSegnet model weight can be downloaded from [here](https://drive.google.com/file/d/1-6Y_cnkerKEtaLHb8L1VnPQkM8GarTiV/view?usp=sharing)

## Segmented Masks
Here are some of the result of NucleiSegnet from the [MultiOrgan Dataset](https://nucleisegmentationbenchmark.weebly.com/dataset.html)

| Organ    | Image  |    Ground Truth | Segmented |    Overlaid | 
| ----------- | ----------- | ----------- | ----------- | ----------- |
|   Liver |  <img src="sample_images/TCGA-21-5784-01Z-00-DX1.jpg" width="100" height="100">  |                                                       <img src="sample_images/TCGA-21-5784-01Z-00-DX1.png" width="100" height="100"> |                                                    <img src="sample_images/TCGA-21-5784-01Z-00-DX1_seg.png" width="100" height="100">    |                                                       <img src="sample_images/TCGA-21-5784-01Z-00-DX1_over.png" width="100" height="100"> |
|   Kidney |  <img src="sample_images/TCGA-B0-5710-01Z-00-DX1.jpg" width="100" height="100">  |                                                       <img src="sample_images/TCGA-B0-5710-01Z-00-DX1.png" width="100" height="100"> |                                                    <img src="sample_images/TCGA-B0-5710-01Z-00-DX1_seg.png" width="100" height="100">    |                                                       <img src="sample_images/TCGA-B0-5710-01Z-00-DX1_over.png" width="100" height="100"> |
|   Bladder |  <img src="sample_images/TCGA-G2-A2EK-01A-02-TSB.jpg" width="100" height="100">  |                                                       <img src="sample_images/TCGA-G2-A2EK-01A-02-TSB.png" width="100" height="100"> |                                                    <img src="sample_images/TCGA-G2-A2EK-01A-02-TSB_seg.png" width="100" height="100">    |                                                       <img src="sample_images/TCGA-G2-A2EK-01A-02-TSB_over.png" width="100" height="100"> |






