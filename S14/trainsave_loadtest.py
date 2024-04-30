import os
import torch
import torchvision
from torchvision import datasets, transforms, utils
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
import pandas as pd
import seaborn as sn
import torch.nn as nn
import torch.nn.functional as F
# from IPython.core.display import display
import misclas_helper
import gradcam_helper
import lightningmodel
from misclas_helper import display_cifar_misclassified_data
from gradcam_helper import display_gradcam_output
from misclas_helper import get_misclassified_data2
from misclas_helper import classify_images
from lightningmodel import LitResnet
#ref : https://pytorch-lightning.readthedocs.io/en/1.2.10/common/weights_loading.html
from pytorch_lightning.callbacks import ModelCheckpoint

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck', 'NotApplicable')

inv_normalize = transforms.Normalize(
  mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
  std=[1/0.23, 1/0.23, 1/0.23]
)

def ts_lt( # Train and Save Vs Load and Test
          save1_or_load0, # decision maker for training Vs testing
          Epochs = 1, # argument for training
          wt_fname = "/content/weights.ckpt" # argument for testing
          ): 
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='/content/',
        filename='weights_{epoch:02d}_{val_acc:.2f}',
        save_top_k=3,
        mode='max',
    )

    trainer = Trainer(
      max_epochs=Epochs, #26
      accelerator="auto",
      devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
      logger=CSVLogger(save_dir="logs/"),
      callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10), checkpoint_callback],
    )
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

    if save1_or_load0 == True:
      model = LitResnet(lr=0.05)
      checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='/content/',
        filename='weights_{epoch:02d}_{val_acc:.2f}',
        save_top_k=3,
        mode='max',
      )
      trainer.fit(model, cifar10_dm)
    else:
      model = LitResnet(lr=0.05).load_from_checkpoint(wt_fname)

    trainer.test(model, datamodule=cifar10_dm)
    
    return (device, model, trainer)