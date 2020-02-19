import numpy as np
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
from imutils import face_utils


def load_image(path):
    return cv2.imread(path)


"""Angry, Afraid, Disgusted, Happy, Neutral, Sad, Surprised """
emotion = { 0:"ANS", 1:"AFS", 2:"DIS", 3:"HAS", 4:"NES", 5:"SAS", 6:"SUS" }


def list_files(dir):
    imgs = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if ".JPG" in name:
                for key in emotion:
                    if emotion[key] in name:
                        imgs.append(os.path.join(root, name))
    return imgs


def reshape_data(input_data):
    nsamples, nx, ny, depth = input_data.shape
    return input_data.reshape((nsamples, nx*ny*depth))


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


KDEF_PATH = '.' + os.path.sep + 'KDEF_and_AKDEF' + os.path.sep + 'KDEF' + os.path.sep
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')

image_paths = list_files(KDEF_PATH)
print(len(image_paths))

imgs = []
emotions = []
cnt = 0
faces = []

try:
    imgs = np.load('imgs_128x128_emotion.npy')
    emotions = np.load('imgs_128x128_emotion_labels.npy')
    faces = np.load('faces_emotion.npy')
except:
    print('Error loading files')

if len(imgs) == 0:
    for image_path in image_paths:
        image_directory, image_name = os.path.split(image_path)

        for key in emotion:
            if emotion[key] in image_name:
                img = load_image(image_path)

                cnt += 1
                print(cnt)
                rects = detector(img, 1)
                for (i, rect) in enumerate(rects):
                    (x, y, w, h) = face_utils.rect_to_bb(rect)
                    shape = predictor(img, rect)
                    shape = shape_to_np(shape)
                    faces.append(shape)
                    if x < 0 or y < 0 or w < 0 or h < 0:
                        continue

                    img = img[y:y + h, x:x + w]
                    img = cv2.resize(img, (128, 128))

                    emotions.append(key)
                    imgs.append(img)

np.save('imgs_128x128_emotion_labels', emotions)
np.save('imgs_128x128_emotion', imgs)
np.save('faces_emotion', faces)

imgs = np.array(imgs)
emotion = np.array(emotions)
faces = np.array(faces)

counter=collections.Counter(emotions)
print(counter)

imgs = imgs / 255.0
print(len(emotion))
print(len(imgs))


X_train, X_test, y_train, y_test = train_test_split(imgs, emotion, test_size=0.25, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()

"""Prvi Conv2d + Relu """
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (128,128,3)))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

"""Drugi Conv2d + Relu """
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

"""Treci Conv2d + Relu """
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
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
model.add(Dense(7, activation = "softmax"))


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
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train,y_train, batch_size=32),
                              epochs = 50, validation_data = (X_test,y_test),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // 32
                              , callbacks=[learning_rate_reduction])

dump(model, '../models/cnn_emotion.joblib')


