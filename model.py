#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:37:28 2017

@author: macbook

## Using original code : NVIDIA ,generator ( center, left,right, flip), samplesize*6

"""
import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
plt.switch_backend('agg')
            
image_paths = []
measurements = []

with open('./data/driving_log.csv') as csvfile:    
    reader = csv.reader(csvfile)
    for line in reader:
        if float(line[6]) < 0.1 :
            continue
        center_path = './data/IMG/'+line[0].split('/')[-1]
        left_path = './data/IMG/'+line[1].split('/')[-1]
        right_path = './data/IMG/'+line[2].split('/')[-1]
        
        center_angle = float(line[3])
        left_angle = center_angle + 0.25
        right_angle = center_angle - 0.25
        
        image_paths.append(center_path)
        measurements.append(center_angle)
        
        image_paths.append(left_path)
        measurements.append(left_angle)
       
        
        image_paths.append(right_path)
        measurements.append(right_angle)


image_paths = np.array(image_paths)
measurements = np.array(measurements)
print('Before:', image_paths.shape, measurements.shape)

# print a histogram to see which steering angle ranges are most overrepresented
num_bins = 25
avg_samples_per_bin = len(measurements)/num_bins
hist, bins = np.histogram(measurements, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(measurements), np.max(measurements)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()

# determine keep probability for each bin: if below avg_samples_per_bin, keep all; otherwise keep prob is proportional
# to number of samples above the average, so as to bring the number of samples for that bin down to the average
keep_probs = []
target = avg_samples_per_bin * .5
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))
remove_list = []
for i in range(len(measurements)):
    for j in range(num_bins):
        if measurements[i] > bins[j] and measurements[i] <= bins[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
image_paths = np.delete(image_paths, remove_list, axis=0)
measurements = np.delete(measurements, remove_list)

# print histogram again to show more even distribution of steering measurements
hist, bins = np.histogram(measurements, num_bins)
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(measurements), np.max(measurements)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()

print('After:', image_paths.shape, measurements.shape)

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
samples = list(zip(image_paths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

        
def generator(samples, batch_size=32):
    """
    Generate the required image_paths and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            #for image,angle in batch_samples:
            for path,measurement in batch_samples:
         
                image = cv2.imread(path)                   
                angle = float(measurement)
                    
                images.append(image)
                angles.append(angle)
                
                #augmented_images, augmented_measurements
                images.append(cv2.flip(image,1))
                angles.append(angle*-1.0)
                
       
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

  
    
#augmented_images, augmented_measurements = [],[]
#for image,measurement in zip(images,measurements):
#    augmented_images.append(image)
#    augmented_measurements.append(measurement)
#    augmented_images.append(cv2.flip(image,1))
#   augmented_measurements.append(measurement*-1.0)
    



from keras.models import Sequential
from keras.layers import Dense,Flatten,Lambda,Cropping2D,ELU,Dropout
from keras.layers import Convolution2D
#from keras.layers import MaxPooling2D
#from keras.layers import pooling



train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))


###commaai
#model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
#model.add(ELU())
#model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
#model.add(ELU())
#model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
#model.add(Flatten())
#model.add(Dropout(.2))
#model.add(ELU())
#model.add(Dense(512))
#model.add(Dropout(.5))
#model.add(ELU())
#model.add(Dense(1))


###NVDIA
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#####LENET
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))

    
model.compile(loss='mse',optimizer='adam')
#model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*2, validation_data=validation_generator, nb_val_samples=len(validation_samples)*2, nb_epoch=3, verbose =1)

model.save('model.h5')

print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()