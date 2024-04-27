# gradioMisClassGradCAMimageInputter
import os
import math
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import datasets, transforms, utils
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import gradio as gr
import misclas_helper
import gradcam_helper
import lightningmodel
from misclas_helper import display_cifar_misclassified_data
from gradcam_helper import display_gradcam_output
from misclas_helper import get_misclassified_data2
from lightningmodel import LitResnet

fileName = None

targets = None
device = torch.device("cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

model = LitResnet(lr=0.05).load_from_checkpoint("weights_92.ckpt")

device = torch.device("cpu")

# Denormalize the data using test mean and std deviation
inv_normalize = transforms.Normalize(
    mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
    std=[1/0.23, 1/0.23, 1/0.23]
)

# Get the misclassified data from test dataset
misclassified_data = get_misclassified_data2(model, device, 20)

def hello(DoYouWantToShowMisClassifiedImages, HowManyImages):
  if(DoYouWantToShowMisClassifiedImages.lower() == "yes"):
    fileName = misclas_helper.display_cifar_misclassified_data(misclassified_data, classes, inv_normalize, number_of_samples=HowManyImages)
    return Image.open(fileName)
  else:
    return None
misClass_demo = gr.Interface(
    fn = hello,
    inputs=['text', gr.Slider(0, 20, step=5)],
    outputs=['image'],
    title="Misclasseified Images",
    description="If your answer to the question DoYouWantToShowMisClassifiedImages is yes, then only it works.",
)


############


def inference(DoYouWantToShowGradCAMMedImages, HowManyImages, WhichLayer, transparency):
  if(DoYouWantToShowGradCAMMedImages.lower() == "yes"):
    if(WhichLayer == -1):
      target_layers = [model.model.resNetLayer2Part2[-1]]
    elif(WhichLayer == -2):
      target_layers = [model.model.resNetLayer2Part1[-1]]
    elif(WhichLayer == -3):
      target_layers = [model.model.Layer3[-1]]
    fileName = gradcam_helper.display_gradcam_output(misclassified_data, classes, inv_normalize, model.model, target_layers, targets, number_of_samples=HowManyImages, transparency=0.70)
    return Image.open(fileName)

gradCAM_demo = gr.Interface(
    fn=inference,
        #DoYouWantToShowGradCAMMedImages, HowManyImages, WhichLayer, transparency
    inputs=['text', gr.Slider(0, 20, step=5), gr.Slider(-3, -1, value = -1, step=1), gr.Slider(0, 1, value = 0.7, label = "Overall Opacity of the Overlay")],
    outputs=['image'],
    title="GradCammd Images",
    description="If your answer to the question DoYouWantToShowGradCAMMedImages is yes, then only it works.",
)


############

def ImageInputter(img1, img2, img3, img4, img5, img6, img7, img8, img9, img10):
  return img1, img2, img3, img4, img5, img6, img7, img8, img9, img10

imageInputter_demo = gr.Interface(
    ImageInputter,
    [
        "image","image","image","image","image","image","image","image","image","image"
    ],
    [
        "image","image","image","image","image","image","image","image","image","image"
    ],
    examples=[
        ["bird.jpg", "car.jpg", "cat.jpg"],
        ["deer.jpg", "dog.jpg", "frog.jpg"],
        ["horse.jpg", "plane.jpg", "ship.jpg"],
        [None, "truck.jpg", None],
    ],
    title="Max 10 images input",
    description="Here's a sample image inputter. Allows you to feed in 10 images and display them. You may drag and drop images from bottom examples to the input feeders",
)


############


demo = gr.TabbedInterface(
    interface_list = [misClass_demo, gradCAM_demo, imageInputter_demo],
    tab_names = ["MisClassified Images", "GradCAMMed Images", "10 images inputter"]
)

demo.launch(debug=True)