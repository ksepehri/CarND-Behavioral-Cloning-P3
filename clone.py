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
correction_factor = 0.2
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = 'my_data/IMG/' + filename
		# print(current_path)
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		if i == 1:
			measurement += correction_factor
		if i == 2:
			measurement -= correction_factor
		measurements.append(measurement)
		#flip image and measurement and add to data as well
		image_flipped = np.fliplr(image)
		images.append(image_flipped)
		measurement_flipped = -measurement
		measurements.append(measurement_flipped)

print('Number of images: {}'.format(len(images)))
print('Number of measurements: {}'.format(len(measurements)))


X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:(x /255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
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
