# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/tsc1.png "Visualization"
[image2]: ./examples/tsc2.png "Grayscaling"
[image3]: ./examples/tsc3.png "GermanTrafficSign"


---

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a histogram showing how the data set's classes are dristributed.

![alt text][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the image data because this typically helps the networks to learn faster since gradients act uniformly for each color channel.

As a second step, I decided to convert the images to grayscale because the color information on traffic sign would not help detect sign's objects and also lower calculation complexity as well.
I tried both preprocessing on the test set but there are not significant difference in terms of accuracy between normalized + grayscaled images and only grayscaled images. 

As a last step, I considered to generate additional data because there are big difference between the number of classes.
However, I leave room for future improvement because only grayscaled images and modified CNN model shows reasonable performance beyond 0.93.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description								| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscaled image						| 
| Convolution 5x5x16   	| 1x1 stride, same padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5x64   	| 1x1 stride, same padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 					|
| Fully connected		| input 1,600, output 1,024 					|
| RELU					|												|
| Dropout				| 0.5											|
| Fully connected		| input 1,024, output 1,024 					|
| RELU					|												|
| Dropout				| 0.5											|
| Fully connected		| input 1,024, output 512 						|
| RELU					|												|
| Dropout				| 0.5											|
| Fully connected		| input 512, output 43 							|
| Softmax				| softmax_cross_entropy_with_logits				|
|						|												|
|						|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam Optimizer for gradient descent and the below hyperparameters. 

| Training         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Optimizer        		| Adam Optimizer								| 
| Batch Size   			| 256 											|
| Number of Epochs		| 60											|
| Learning Rate	      	| 0.001 										|
| Dropout   			| 0.5 											|
|						|												|
|						|												|


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 96.2%
* test set accuracy of 95.3%

I started training test sets by using default LeNet network without any modification. The result of first training was below 0.9.
My approach for finding a solution is making neural network deeper and wider based on LeNet architecture.

Although hand written numbers have 10 digits, traffic signs have 43 classes which requiring more complex neural network.
For this reason, I modified convolutional layer's depth from 16 to 64 and add one more fully connected layer which is 10 times wider than LeNet's.
I trained modified versions of LeNet serveral times and I found that there are some difference between validation set's and test set's accuracy.
So, I added dropout step between fully connected layers in order to lower the gap from overftting.

However, test set's accuracy was slightly above 0.93 so that I need to consider parameter tunings to improve the result.
I changed weight and bias initialization methods from truncated normalization to Xavier's initialization.
Networks initialized with Xavier achieved substantially quicker convergence and higher accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are randomly selected five German traffic signs that I found on the web:

![alt text][image3]

The all image might not be difficult to classify because I pre-processed whole images in the same way when I trained model.
I made them have (32, 32, 1) shape and converted grayscaled color right before testing. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.
This compares favorably to the accuracy on the new traffic signs.
As you can see the above images, I assume that German traffic sign images show impeccable quality.
Those images clearly show traffic sign without any obstacle like shadow or other objects.
Instead of randomly picked 5 images, I tested all 26,640 traffic sign images and I got 99.7% accuracy. 

Here are the results of the prediction:

| Image				        |     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Wild animals crossing 	| Wild animals crossing							| 
| Road work     			| Road work										|
| Yield						| Yield											|
| Bicycles crossing    		| Bicycles crossing				 				|
| Traffic signals			| Traffic signals      							|
 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the last image among 5 images, the model is 100% sure that this is a traffic signals (probability of 1.0), and the image does contain a traffic signals. The top five soft max probabilities and logits were

| Probability (logits) 	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0 (45.022) 			| Traffic Signs									| 
| 0.0 (8.354)			| General Caution								|
| 0.0 (4.526)			| Pedestrians									|
| 0.0 (4.244)  			| Dangerous curve to the right	 				|
| 0.0 (2.937)			| Road narrows on the right						|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


