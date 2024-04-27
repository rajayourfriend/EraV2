---
title: EraV2s13 Raj
emoji: 📈
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 4.28.3
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## For Assignment 13


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
