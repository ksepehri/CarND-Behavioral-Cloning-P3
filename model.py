import os
import csv
from random import shuffle

samples = []
with open('my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
print('Samples: {}'.format(len(samples)))
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('Training Samples: {}'.format(len(train_samples)))
print('Validation Samples: {}'.format(len(validation_samples)))

import cv2
import numpy as np
import sklearn

correction_factor = 0.2

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                
                # use all images from car(left, center, right)
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = 'my_data/IMG/' + filename
                    
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    
                    #apply a steering correction factor for left and right images 
                    if i == 1:
                        measurement += correction_factor
                    if i == 2:
                        measurement -= correction_factor
                    angles.append(measurement)
                    
                    #flip image and measurement and add to data as well
                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)
                    measurement_flipped = -measurement
                    angles.append(measurement_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:(x /255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24, 5, 5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36, 5, 5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48, 5, 5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64, 3, 3,activation="relu"))
model.add(Convolution2D(64, 3, 3,activation="relu"))
model.add(MaxPooling2D((1, 1)))
model.add(Flatten())
model.add(Dropout(.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

epochs = 1

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples*6), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=epochs, verbose=1)

import matplotlib.pyplot as plt


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
print('Model Saved')