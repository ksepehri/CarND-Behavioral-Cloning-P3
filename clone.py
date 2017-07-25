import csv
import cv2
import numpy as np
# Basic load code copied from lecture video

lines = []
with open('my_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

print('Number of lines: {}'.format(len(lines)))

images = []
measurements = []

for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = 'my_data/IMG/' + filename
	# print(current_path)
	image = cv2.imread(current_path)
	images.append(image)
	measurement = line[3]
	measurements.append(measurement)

print('Number of images: {}'.format(len(images)))
print('Number of measurements: {}'.format(len(measurements)))


X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:(x /255.0) - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(32, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')

exit()
