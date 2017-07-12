#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. NVIDIA model architecture has been employed

I started with LENET, then tried NVIDIA and finally commai.
But i ended up using NVIDIA.

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 167-171) 

The model includes RELU layers to introduce nonlinearity (code line 172 and 173), and the data is normalized in the model using a Keras lambda layer (code line 147). 

####2. Attempts to reduce overfitting in the model

Both training and validation losses decreased for each of my three epochs. So i didn't use dropouts for overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 189).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a try LENET that was taught in the course with the sample data provided. Car was driving well in autonomous mode. I then collected my own data from simulation and tried with LENET. 

The car was driving off the edge. So i tried NVIDIA and later commai. Both trained well but at some turns car was driving off the edges.

Then I discarded my previous data (that i collected) and tried to capture proper data. This time i drove in the forward direction as well as reverse directions. I collected more data to make sure i was going towards the edge and the moved back to the center (for both left and right turns)


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I plotted a distribution of angles.
I reduced the angles with zero distributions and made sure the larger angles had significant distributions compared to lower angles.


Using NVIDIA architecuture, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 167 to 176) consisted of a convolution neural network with the following layers and layer sizes ...

Layer 0: Lambda, Normalisation to range -0.5, 0.5 (1./255 -1)

Layer 1: Cropping, to remove 50 from top and 20 from  bottom

Layer 2: Convolution with strides=(2,2),kernel 5x5 and output shape 245x5x5, with relu activation

Layer 3: Convolution with strides=(2,2),kernel 5x5 and output shape 36x5x5, with relu activation

Layer 4: Convolution with strides=(2,2),kernel 5x5 and output shape 48x5x5, with relu activation

Layer 5: Convolution with kernel 3x3 and output shape 64x3x3, with relu activation

Layer 6: Convolution with kernel 3x3 and output shape 64x3x3, with relu activation

Layer 7 : flatten 1152 output

Layer 8: Fully Connected with 100 outputs

Layer 9: Fully Connected with 50 outputs

Layer 10: Fully Connected with 10 outputs

Output Layer : Fully Connected with 1 output value for the steering angle





![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded laps (forward and opposite directions) on track one using center lane driving. 
![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to center

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had 23,938 number of data points. I then converted them into images and angles for left, right and center cameras, while filtering out data points with speed less then 0.1. I ended up with around 70k images. After redistributing the number of samples to represent most anlges,  ended up with 21115 images.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
I used generator to load the actual images in batches. I augemented data in the generator itself. I flipped images to enhance data.

The ideal number of epochs was 3 as evidenced by both training and validation errors simultaneously declining. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Miscellaneous:
I used video_ocv.py to combine images into video since video.py was giving me issues.
video_ocv.py is attached.
Drive.py --didnt do much preprocessing in model.py hence didnt modify drive.py very much.

