import csv
#import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
import matplotlib.image as mpimg

lines = []

#read rows of csv file with image names and steering angles
with open('./TrainingOwn/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#remove header line
del lines[0]

#define split of input dataset into train and validation set. 
train_lines, validation_lines = train_test_split(lines, test_size=0.3)


# create adjusted steering measurements for the side camera images
correction = 0.0 # this is a parameter to tune

#define generator to manage huge set of data in memory
def generator(gen_lines, batch_size=32):
    num_lines = len(gen_lines)
    while 1:
        #shuffle data for training
        shuffle(gen_lines)
        #process images per batch size
        for offset in range(0, num_lines, batch_size):
            batch_lines = gen_lines[offset:offset+batch_size]
            images = []
            angles = []
            for batch_line in batch_lines:
                #processing only center images
                for i in range(1):
                    source_path = batch_line[i]
                    #source_path = batch_line[0]
                    filename = source_path.split('\\')[-1]
                    current_path = './TrainingOwn/IMG/' + filename
                    #image = cv2.imread(current_path)
                    image = mpimg.imread(current_path)
                    angle = float(batch_line[3])
                    
                    #if i == 1:
                    #    angle = angle + correction
                    #elif i == 2:
                    #    angle = angle - correction
                    
                    images.append(image)
                    angles.append(angle)
                    #image_flipped = cv2.flip(image,1)
                    #flip image to get more data for training
                    image_flipped = np.fliplr(image)
                    angle_flipped = angle*-1.0
                    images.append(image_flipped)
                    angles.append(angle_flipped)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

#import KERAS for training
            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU, Activation
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

#run generator on training and validation set
train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(validation_lines, batch_size=32)

input_shape=(160,320,3)

#my network
model = Sequential()
#normalize input
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
#remove top and bottom part of the image as they are not relevant for the training
model.add(Cropping2D(cropping=((70,25),(0,0)))
#1st convolution layer with kernel size 5x5, stride 2x2, depth 24, and with RELU activation
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
#2nd convolution layer with kernel size 5x5, stride 2x2, depth 36, and with RELU activation
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
#3rd convolution layer with kernel size 5x5, stride 2x2, depth 48, and with RELU activation
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
#4th convolution layer with kernel size 3x3, stride 1, depth 64, and with RELU activation
model.add(Convolution2D(64,3,3,activation="relu"))
#5th convolution layer with kernel size 5x5, stride 1, depth 64, and with RELU activation
model.add(Convolution2D(64,3,3,activation="relu"))
#flatten layer
model.add(Flatten())
#1st fully connected layer with output size 100
model.add(Dense(100))
#2nd fully connected layer with output size 50
model.add(Dense(50))
#3rd fully connected layer with output size 10
model.add(Dense(10))
#final fully connected layer with output size 1 as we have 1 label
model.add(Dense(1))

#configuring learning process with 'adam' optimizer and 'mse' loss function
model.compile(loss='mse', optimizer='adam')
#run training
model.fit_generator(train_generator, samples_per_epoch= 6*len(train_lines), validation_data=validation_generator, nb_val_samples=6*len(validation_lines), nb_epoch=5)

#save model
model.save('model.h5')
