# Yes - This is important predecessor1 for gradioMisClassGradCAM
import numpy as np
import gradio as gr
from PIL import Image
import torch
import torchvision
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import datasets, transforms, utils
import math
import os
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import datasets, transforms, utils
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.datamodules import CIFAR10DataModule




def get_misclassified_data2(model, device, test_loader, count):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """

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

