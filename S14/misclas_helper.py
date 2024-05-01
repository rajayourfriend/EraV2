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
import torch.nn as nn
import torch.nn.functional as F
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


def get_misclassified_data2(model, device, count):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """

    PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
    BATCH_SIZE = 256 if torch.cuda.is_available() else 64
    NUM_WORKERS = int(os.cpu_count() / 2)


    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir=PATH_DATASETS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    cifar10_dm.prepare_data()
    cifar10_dm.setup()
    test_loader = cifar10_dm.test_dataloader()

    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()

    # List to store misclassified Images
    misclassified_data = []

    # Reset the gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:

            # Migrate the data to the device
            data, target = data.to(device), target.to(device)

            # Extract single image, label from the batch
            for image, label in zip(data, target):

                # Add batch dimension to the image
                image = image.unsqueeze(0)

                # Get the model prediction on the image
                output = model(image)

                # Convert the output from one-hot encoding to a value
                pred = output.argmax(dim=1, keepdim=True)

                # If prediction is incorrect, append the data
                if pred != label:
                    misclassified_data.append((image, label, pred))

                if len(misclassified_data) > count :
                  break
    return misclassified_data


# Yes - This is important predecessor2 for gradioMisClass

def display_cifar_misclassified_data(data: list,
                                     classes: list[str],
                                     inv_normalize: transforms.Normalize,
                                     number_of_samples: int = 10):
    """
    Function to plot images with labels
    :param data: List[Tuple(image, label)]
    :param classes: Name of classes in the dataset
    :param inv_normalize: Mean and Standard deviation values of the dataset
    :param number_of_samples: Number of images to print
    """
    fig = plt.figure(figsize=(10, 10))
    img = None
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        img = data[i][0].squeeze().to('cpu')
        img = inv_normalize(img)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.xticks([])
        plt.yticks([])
    plt.savefig('imshow_output_misclas.png')
    return 'imshow_output_misclas.png'

# Plot the misclassified data


def crop_image_pil2(image): #Crop image with 1:1 output aspect ratio

    image = Image.fromarray(image)
    print("image type = ", type(Image)) 
    width, height = image.size
    if width == height:
        return image
    offset  = int(abs(height-width)/2) 
    if width>height:
        image = image.crop([offset,0,width-offset,height])
    else:
        image = image.crop([0,offset,width,height-offset]) 
    return image

def resize_image_pil2(image, new_width, new_height):
    # Convert to PIL image
    img = crop_image_pil2(image)
    img = Image.fromarray(np.array(img))
    # Get original size
    width, height = img.size

    # Calculate scale
    width_scale = new_width / width    # RAJA see if this can be deleted
    height_scale = new_height / height # RAJA see if this can be deleted
    # Resize
    # resized = img.resize((int(width*width_scale), int(height*height_scale)), Image.NEAREST)
    resized = img.resize((32, 32), Image.NEAREST)
    # Crop to exact size
    return resized
    
def classify_images(list_images, model, device):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )
    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()
    # List to store misclassified Images
    classified_data = []

    # Reset the gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for image in list_images:
          print("image type = ", type(image))
          orig_image = image
          if(image is None):
            pred = 10 #This entry indicates none in classes, empty string
          else:
            print("before resize image shape = ", image.shape)
            image = resize_image_pil2(image, 32, 32)
            print("after resize image shape = ", image.shape)
            image = np.asarray(image)
            print("numpy image dtype = ", image.dtype)
            print("before transpose image shape = ", image.shape)
            image = np.transpose(image, (2, 1, 0))
            print("after transpose image shape = ", image.shape)
            image = torch.from_numpy(image).float()
            print("before test_transforms image shape = ", image.shape)
            image = test_transforms(image)
            print("after test_transforms image shape = ", image.shape)

            image = image.unsqueeze(0)
            print("after squeeze image shape = ", image.shape)

            # Get the model prediction on the image
            output = model(image)

            # Convert the output from one-hot encoding to a value
            pred = output.argmax(dim=1, keepdim=True)

          classified_data.append((orig_image, pred))

    return classified_data