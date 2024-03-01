
# PART-1

My results in excel file does not match with the one available in the video, hence not submitting it now.

# PART-2

### How many layers ?
Answer : More the number of layers, better will be the performance. Deep neural networks are expected to contain more number of layers. The number of layers should be such that receptive field at the last layer atleast the image size or more.

### MaxPooling 
Answer : Highest amplitude from input among set of pixels, only will go to output without any computation. Hence no additional parameters to be learnt. It does not alter the number of channels, but the output dimension will be reduced.

### 1x1 Convolutions 
Answer : Superset of linear network or fully connected network. Unlike maxpool / minpool, learnable parameters exists that become the reason for crteria for output derived from input.

### 3x3 Convolutions 
Answer : Most optimized kernel size being 3x3 will filter the required features from the image. Spatial information is not lost and the kernel would extract the features such as edges & gradients, textures & patterns, parts of objects and objects.

### Receptive Field 
Answer : This measures the visibility area of image from a specific layer or on the whole, respectively local RF and global RF.

### SoftMax
Answer : It projects the prediction with a comparatively higher amplitude so that prediction will be affirmed without any confusion. In other words, interpretability of the network will be very distinct. The sum of all the output of softmax would be 1.

### Learning Rate
Answer : It specifies with how much speed the movement of next point of prediction need to traverse from the present point.

### Kernels and how do we decide the number of kernels?
Answer : Feature extractors are called kernels. Higher the number of kernels, better the performance is. There are two architectures related to number of kernels in a network in comparison with adjacent layers. 1) Cake shape 2) X-mas tree shape.
1) Cake shape : Few consecutive layers would have the same number of kernels in the adjacent layers. Afterwards number of kernels would be altered (increased in succeeding layers) and maintained same for few adjacent layers and this phenomenon will be repeated in further layers.
2) X-mas tree shape : A steady increase in number of kernels will be there in the adjacent layers. After that with maxpool / minpool would be placed that would reduce the image size to be reduced in a big way. Next number of kernels would be increased steadily in the adjacent layers. With further placement maxpool/minpool this phenomenon would be repeated.

### Batch Normalization 
Answer : A batch of input images are considered for computation to apply normalization. 
Image Normalization
Answer : The whole set of images in the dataset is considered for computation to apply normalization.

### Position of MaxPooling 
Answer : After few layers of increased number of kernels, a maxpool shall be placed. A maxpool layer shall be kept as far away from the last layer, looking from reverse. Usage of more than one layer of MP is avoided in continuum in a network.

### Concept of Transition Layers 
Answer : A transition layer is used to control the complexity of the model. It reduces the number of channels by using a 1x1 convolution.

### Position of Transition Layer
Answer : No knowledge on this topic.

### DropOut
Answer : The percentage of input pixels / intermittent computation output to be dropped from going to arrive at the prediction for learning / computation.

### When do we introduce DropOut, or when do we know we have some overfitting ?
Answer : We can use Dropout when training accuracy is more than validation accuracy. That means training is good, but not validation. The main indicator is that even though the number of epochs are increased tremendously, still the training accuracy does not improve significantly and thus validation accuracy is poor.

### The distance of MaxPooling from Prediction
Answer : As far away. In general, 2 to 4 layers back from last layer.

### The distance of Batch Normalization from Prediction 
Answer : It shall be done to almost every layer, just before reaching the last layer. That means BN should not be present for the last layer.

### When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
Answer :  No knowledge on this topic.

### How do we know our network is not going well, comparatively, very early
Answer : Take very few number of epochs such as 6, comparing with previous curve, it is in the downside or fallside or drooping way.

### Batch Size, and Effects of batch size
Answer : More the batch size, better the network performance is. But after some point, drooping characteristics would be observed.


# PART-3

## 1. Import external libraries
This section does import of external libraries that are used in the present file ToSubmit_04_EraV2_S6.ipynb, such as pytorch, etc.


## 2. Convolutional Neural Network (model) architecture
This section does the preparation of model alias network that has several layers such as convolution, maxpool, batch normalization, dropout, etc. In the function forward, input x is fed to conv1 layer first and then passed through an activation layer of relu. The output is fed to batch normalization process. After passing another set of conv (2), relu, batch_norm, the output is fed to a maxpool layer. This is followed in the same way to reach another maxpool layer. After that output is fed to conv (7) layer. From this, the output is fed to pytorch view function that does reshape of dimension to 1x10 that is the final output layer.


## 3. Display summary of model
This section gets and displays the summary of the model along with the number of parameters used in the model, layerwise. Total number of parameters are 13422.

## 4. Preparation of dataset
This section prepares the dataset of MNIST (Modified National Institute of Standards and Technology) from pytorch that is nothing but images of handwritten numbers 0 to 9.

## 5. Train and Test Functions
This section comprises train and test functions that does the training and validation required by the model to get trained and finally get validated.

## 6. Run the model with a device and an optimizer
This section contains code to run the model to get trained using a device that is GPU and an optimizer that is SGD(Stochastic Gradient Descent).

On epoch 12, validation accuracy is 99.43 %  
On epoch 17, validation accuracy is 99.50 %
