import numpy as np
import glob
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
import collections
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from joblib import dump
import dlib
import time
from imutils import face_utils

def load_image(path):
    return cv2.imread(path)


def reshape_data(input_data):
    nsamples, nx, ny, depth = input_data.shape
    return input_data.reshape((nsamples, nx*ny*depth))


UTK_PATH = '.' + os.path.sep + 'UTKface_inthewild' + os.path.sep + 'part2' + os.path.sep
detector = dlib.get_frontal_face_detector()

age = []
gender = []
race = []
imgs = []

imgs1 = []
imgs2 = []
age1 = []
age2 = []

ages = {1:[1,5],
        2:[6,10],
        3:[11,15],
        4:[16,19],
        5:[20,25],
        6:[26,32],
        7:[33,40],
        8:[41,50],
        9:[51,60],
        10:[61,70],
        11:[71,80],
        12:[81,1000]
        }

start = time.time()
if len(imgs) == 0:
    cnt = 0

try:
    imgs2 = np.load('imgs_64x64_p2.npy')
    age2 = np.load('imgs_64x64_age_p2.npy')
    imgs1 = np.load('imgs_64x64_p1.npy')
    age1 = np.load('imgs_64x64_age_p1.npy')
except:
    print('Failed to load face images')

if len(age2) == 0:
    cnt = 0

    for image_path in glob.glob(UTK_PATH + "*.jpg"):
        image_directory, image_name = os.path.split(image_path)
        age_var = image_name.split('_')[0]
        gender_var = image_name.split('_')[1]
        race_var = image_name.split('_')[2]
        img = load_image(image_path)

        cnt += 1
        print(cnt)

        rects = detector(img, 1)
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            if i == 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if x < 0 or y < 0 or w < 0 or h < 0:
                    continue

                img = img[y:y+h,x:x+w]
                img = cv2.resize(img, (64, 64))

                if gender_var == '0' or gender_var == '1':
                    for key in ages:
                        if int(age_var) >= ages[key][0] and int(age_var) <= ages[key][1]:
                            if(race_var == '0' or race_var == '1' or race_var == '2' or race_var == '3' or race_var == '4'):
                                age.append(key)
                                race.append(race_var)
                                gender.append(gender_var)
                                imgs.append(img)
            else:
                continue
    np.save('imgs_64x64_p2',imgs)
    np.save('imgs_64x64_gender_p2',gender)
    np.save('imgs_64x64_age_p2', age)
    np.save('imgs_64x64_race_p2',race)

imgs = np.vstack((imgs1,imgs2))
age = np.concatenate((age1,age2))
imgs = np.array(imgs)
age = np.array(age)

counter=collections.Counter(age)
print(counter)


# Normalize the data
imgs = imgs / 255.0
print(len(age))
print(len(imgs))

X_train, X_test, y_train, y_test = train_test_split(imgs, age, test_size=0.25, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(X_train.shape)

model = Sequential()

"""Prvi Conv2d + Relu """
model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (64,64,3)))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

"""Drugi Conv2d + Relu """
model.add(Conv2D(filters = 192, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(filters = 192, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

"""Treci Conv2d + Relu """
model.add(Conv2D(filters = 192, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(filters = 192, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

"""Cetvrti Conv2d + Relu """
model.add(Conv2D(filters = 192, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(filters = 192, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

"""Peti Conv2d + Relu """
model.add(Conv2D(filters = 192, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(filters = 192, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


"""Prvi FC Layer"""
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))

"""Drugi FC Layer"""
model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))

"""FC Softmax Output Layer"""
model.add(Dense(13, activation = "softmax"))


# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=7,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
# obucavanje neuronske mreze


model.fit_generator(datagen.flow(X_train,y_train, batch_size=128),
                              epochs = 50, validation_data = (X_test,y_test),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // 128
                              ,callbacks=[learning_rate_reduction])


dump(model, './models/cnn_age_p1_p2.joblib')