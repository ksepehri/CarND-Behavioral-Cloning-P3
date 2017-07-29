**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup-media/left_2017_07_28_13_49_53_447.jpg "Left"
[image2]: ./writeup-media/center_2017_07_28_13_49_53_447.jpg "Center"
[image3]: ./writeup-media/right_2017_07_28_13_49_53_447.jpg "Right"

[image4]: ./writeup-media/center_2017_07_28_13_48_54_398.jpg "Recovery Image"
[image5]: ./writeup-media/center_2017_07_28_13_48_54_570.jpg "Recovery Image"
[image6]: ./writeup-media/center_2017_07_28_13_48_54_986.jpg "Recovery Image"

[image7]: ./writeup-media/left_2017_07_28_13_48_36_817.jpg "Flipped Image"
[image8]: ./writeup-media/left_2017_07_28_13_48_36_817_flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeupd.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the [Nvidia network architecture] (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model with 5 convolutional layers going from 5x5 to 3x3 filter sizes and depths between 24 and 64 (model.py lines 68-82)

The model includes RELU layers to introduce nonlinearity (lines 70-75), and the data is normalized in the model using a Keras lambda layer (line 69). 

Max pooling is used after the convolutions to to reduce the size of the input, and allow the neural network to focus on only the most important elements.

Finally the model is flattened and dropout is applied. Dropout is used so that "the network can't rely on any given activation to be present. The network is forced to learn a redundant representation for everything." (Vincent Vanhouke)

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (line 78). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (line 22-58). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving the track counterclockwise, and recovery driving counterclockwise.

I used center, left, and right images. I found that this really helped train the car on center lane driving. 

Additionally I recorded extra data for problem areas.
1. Red and white lines instead of yellow lines
2. Big curve after first red and white lines
3. Bridge
5. Dirt area after bridge

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a model that did not overfit.

My first step was to use a convolution neural network model similar to the Nvidia model. I thought this model might be appropriate because it was proven by them to be helpful for autonomous driving.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it used dropout and max pooling. Then I started getting results that could almost drive the track. There were a few problem areas.

Then I provided extra data for the problem areas and a run of the 2nd track just to generalize the data a bit more.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

1. 24@5x5 Convolution
1. 36@5x5 Convolution
1. 48@5x5 Convolution
1. 64@3x3 Convolution
1. 64@3x3 Convolution
1. 1x1 Max Pooling
1. Flatten
1. 50% Dropout
1. Dense

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving from left, center, and right angle:

![alt text][image1]
![alt text][image2]
![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from a bad position. I would only record the recovery and not driving over to the line in order to only teach the model "good" behavior. These images show what a recovery looks like:

![alt text][image4]

![alt text][image5]

![alt text][image6]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and cropped the image thinking that this would narrow the focus area and help the model generalize. For example, here is an image that has then been flipped:

![alt text][image7]
![alt text][image8]


After the collection process, I had 9,748 number of data points. Since there's 3 images per data point and I added a flipped image(with a negative measurement) for each I ended up with 46,788 images to use.

Finally I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1 for my model and data. Using more epochs resulted in overfitting and the model not being able to recover from corners. I used an adam optimizer so that manually training the learning rate wasn't necessary.
