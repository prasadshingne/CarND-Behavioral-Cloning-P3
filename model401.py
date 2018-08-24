import csv
import cv2
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

lines = []
"""
Retreive lines from driving log
"""
with open('./data02/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None) 
    for line in reader:
        lines.append(line)

"""
Find and combine center, left and right images from the specified path
"""
images = []
measurements = []
count = 0
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data02/IMG/' + filename
        #current_path = './data_02/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        
        if i == 0:
            measurement = float(line[3])
            #print('i=0')
            #time.sleep(5)
        elif i == 1:
            measurement = float(line[3]) + 0.2
            #print('i=1')
            #time.sleep(10)
        else:
            measurement = float(line[3]) - 0.2
            #print('i = 2')
            #time.sleep(5)
        measurements.append(measurement)
    count += 1

"""
Flip images about vertical access and append 
"""
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

"""
Create training arrays 
"""    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import load_model

"""
Model start with prepocessing --> Normalize image, crop 70 and 25 rows 
of pixels from top and bottom of image respectively
"""
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

"""
Implement nVidea Autonomous Car Model-
Model uses convolutional layers to add non-linearity to to the model.
Dropout layers are used to reduce model overfitting.
"""
model.add(Convolution2D(24,(5,5), activation='relu', strides=(2,2)))
model.add(Convolution2D(36,(5,5), activation='relu', strides=(2,2)))
model.add(Convolution2D(48,(5,5), activation='relu', strides=(2,2)))
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(Flatten())
Dropout(.5)
model.add(Dense(100))
Dropout(.1)
model.add(Dense(50))
Dropout(.1)
model.add(Dense(10))
Dropout(.1)
model.add(Dense(1))

"""
Compile and train model - 
Adam optimizer is used
Data is split into 80% training and 20% validation set
Shuffle inputs
Use 3 epochs
Record 
"""
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, verbose=1, epochs=3)

"""
Save model
"""
model.save('model401.h5')

"""
Print the keys contained in the history object
"""
print(history_object.history.keys())

"""
Display model summary
"""
load_model('model401.h5').summary()

"""
Plot the training and validation loss for each epoch
"""
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='lower left')
plt.savefig('fit_history401.png',dpi = 300)
exit()