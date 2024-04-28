## Assignment 13

### What is done and how

First I took Rohan shared input file, S13.ipy that is an S11 reference, where resnet code exists with lots of gradio examples.
Ensured it builds successfully.
Then I replaced the model to my model of S11.
Prepared the code in such a way that on a condition check, the training and save of weights would happen in one path. Otherwise loading the weights, it would perform testing. This weight file is .pth.

Secondly I took Rohan shared input file, cifar10-baseline.ipynb that is an S13 reference, where pytorch lightning code of a working model is present.
Ensured it builds successfully.
Then I replaced the model to my model of S11.
Prepared the code in such a way that on a condition check, the training and save of weights would happen in one path. Otherwise loading the weights, it would perform testing. This weight file is .ckpt.

Thirdly I started working on the gradio with gradcam and made it working for a single image input.
I started working on the gradio with misclassified image display and made it working.
I started working on the gradio with gradcam and made it working for a multiple images taken from cifar 10 misclassified images. 
I started working on the gradio with 10 images inputter and made it working. Here it can accept 1 images and display them. Does not do anything further.
I integrated above said three items such as gradcam for multiple images, misclassified images, 10 images input in gradio. It works successfully.

Fourthly I started working on above working code to modularize it so that few py files will hold major part of code.
I started working on HuggingFace and created / updated required files and it is made to be in working state in HuggingFace.


My spaces app has these features:
1. Asks the user whether he/she wants to see GradCAM images and how many, and from which layer, allow opacity change as well
2. Asks whether he/she wants to view misclassified images, and how many
3. Allow users to upload new images, as well as provide 10 example images

In a tabbed interface, gradio framework is used and available for use from HuggingFace.

HuggingFace https://huggingface.co/spaces/raja5259/eraV2s13_raj
Github https://github.com/rajayourfriend/EraV2/

### Log of Training  

Below is the log of training with pytorch lightning for 26 epochs with 6.6M params and got a test_acc of 91.28%

INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
Files already downloaded and verified
Files already downloaded and verified
INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
INFO:pytorch_lightning.callbacks.model_summary:
  | Name  | Type    | Params
----------------------------------
0 | model | Net_S13 | 6.6 M 
----------------------------------
6.6 M     Trainable params
0         Non-trainable params
6.6 M     Total params
26.293    Total estimated model params size (MB)
Epoch 25: 100%
 197/197 [00:23<00:00,  8.53it/s, loss=0.0844, v_num=3, val_loss=0.261, val_acc=0.916]
INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=26` reached.
Files already downloaded and verified
Files already downloaded and verified
INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing DataLoader 0: 100%
 40/40 [00:03<00:00, 11.30it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │    0.9128999710083008     │
│         test_loss         │    0.2818313539028168     │
└───────────────────────────┴───────────────────────────┘
[{'test_loss': 0.2818313539028168, 'test_acc': 0.9128999710083008}]



