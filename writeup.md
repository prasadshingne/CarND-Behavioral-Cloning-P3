**Behavioral Cloning Project Writeup**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictures/left_2016_12_01_13_31_15_513.jpg "Left"
[image2]: ./pictures/center_2016_12_01_13_31_15_513.jpg "Center"
[image3]: ./pictures/right_2016_12_01_13_31_15_513.jpg "Right"
[image4]: ./pictures/center_2018_08_13_05_56_33_832.jpg "Recovery Image1"
[image5]: ./pictures/center_2018_08_13_05_56_34_421.jpg "Recovery Image2"
[image6]: ./pictures/center_2018_08_13_05_56_35_146.jpg "Recovery Image3"
[image7]: ./pictures/fit_history401.png "Fit History"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model401.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (This is provided by Udacity and I did not modify it)
* model401.h5 containing a trained convolution neural network 
* writeup.md or writeup_report.pdf summarizing the results
* video.mp4 video containig the simulator in autonomous mode using my model401.h5

#### 2. Submission includes functional code
Using the Udacity provided simulator, drive.py file and my model401.h5 the car can be driven autonomously around the first track by executing 
```sh
python drive.py model401.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The nVidia model with convolution neural networks is used. See implementation in model401.py lines 87-100. 

The model includes RELU layers to introduce nonlinearity (code lines 87-91). The data is normalized in the model using a Keras lambda layer (code line 79) and images are cropped (code line 80) to remove the top and bottom in order to have most of the image be the road.

Model summary is as follows - 

|Layer (type)         |        Output Shape      |        Param # |
|:------------------------:|:--------------------:|:--------------:|
|lambda_1 (Lambda)  |       (None, 160, 320, 3)    |      0       |
|cropping2d_1 (Cropping2D)  |  (None, 65, 320, 3)  |      0       |
|conv2d_1 (Conv2D)     |       (None, 31, 158, 24) |      1824    |
|conv2d_2 (Conv2D)     |       (None, 14, 77, 36)  |      21636   | 
|conv2d_3 (Conv2D)     |       (None, 5, 37, 48)   |      43248   |
|conv2d_4 (Conv2D)     |       (None, 3, 35, 64)   |      27712   |
|conv2d_5 (Conv2D)     |       (None, 1, 33, 64)   |      36928   |
|flatten_1 (Flatten)   |       (None, 2112)        |      0       |
|dense_1 (Dense)       |       (None, 100)         |      211300  |
|dense_2 (Dense)       |       (None, 50)          |      5050    |
|dense_3 (Dense)       |       (None, 10)          |      510     |
|dense_4 (Dense)       |       (None, 1)           |      11      |

Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0

More details provided below.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (code lines 94, 96, 98 and 100). Further I keep the training epochs low and split the 
data into training and validation sets. 80% for training and 20% for validation. (code line 112)


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line 112).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the sample data provided by Udacity and additional data from track 1. 
The simulater generates three images: center, left and right; I used all of these for training. I also flipped these images left to right across the vertical axis to augment the data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use the LeNet architecture (not in the submitted code). Then I implemented the nVidia model as instructed. I performed normalization with the Lambda layer
and added a cropping layer. The nVidia architecture is very powerful as it contains a number or convolutional layers with 'relu' to add non-linearity. The only modification I made was adding dropout layers to reduce overfitting.
When I trained on the default data the car ran off the road right after the first turn. Hence I augmented the data set by flipping the images left to right. This improved things but the car was going to the sides too often and off the road 
after the bridge where ther was no railing. I recorded additional three laps roughly on the first track in the opposite direction (right turns). This set included 'recovery driving' as well. I also played with adding dropout layers between convolution layers but 
decided against it as they didn't improve the performance.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (code lines 80-101) consisted of - 

1. Lambda: input(160, 320, 3)
2. Cropping2D: cropping=((70,25),(0,0))

3. Convolution2D: kernal-(5,5), activation=relu, strides = (2,2), filters = 24
4. Convolution2D: kernal-(5,5), activation=relu, strides = (2,2), filters = 36
5. Convolution2D: kernal-(5,5), activation=relu, strides = (2,2), filters = 48
6. Convolution2D: kernal-(3,3), activation=relu, strides = (1,1), filters = 64
7. Convolution2D: kernal-(3,3), activation=relu, strides = (1,1), filters = 64

8. Dense: output=100, activation=linear, input = 2112
9. Dense: output=50, activation=linear, input = 100
10. Dense: output=10, activation=linear, input = 50
11. Dense: output=1, activation=linear, input = 10

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used all the three camaras of all the data available and additionally collected. Here is an example image of center lane driving:

![alt text][image1]
![alt text][image2]
![alt text][image3]

I also recorded roughly three laps while driving in the opposite direction (right) which included scenarios where the vehicle is recovering from the left side and right sides of the road back to center. 
Following figures show a recovery example from the right side of the road to the middle.

![alt text][image4]
![alt text][image5]
![alt text][image6]

I flipped images and steering angles of each image to augment the data set.

Thus I had 64662 number of data points. I then preprocessed this data by normalizing the image and cropping the bottom and top lines of pixels to focus on the road alone (cutting out the hood and the extra horizon)

I randomly shuffled the data set and put 20% of the data into a validation set (12938). 

I used 80% of the sample data for training (51724) the model. The training was done with the adam optimizer for 3 epochs. The following picture shows the training and validaion error history of the model.

![alt text][image7]

After this training the car was able to drive around track one ([video] (https://github.com/prasadshingne/CarND-Behavioral-Cloning-P3/blob/master/videos/track1.mp4)) while being on the road all the time.


#### Note about track two

I did not take any trainig data on track two in the interest of time. However I was curious how the model that had been trained only on track one would perform on track two. To my surprise the car went further than I had expected ([video] (https://github.com/prasadshingne/CarND-Behavioral-Cloning-P3/blob/master/videos/track2_test.mp4)), 
showing that even the current small model is fairly robust. 
