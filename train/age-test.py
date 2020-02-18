from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import pandas as pd
from joblib import dump
import time

from keras_preprocessing.image import ImageDataGenerator

TRAIN_IMAGES_PATH = './data/train'
VALIDATION_IMAGES_PATH = './data/val'

LABELS_PATH = 'fairface_label_train.csv'
VAL_LABELS_PATH = 'fairface_label_val.csv'

ages = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
dftrain = pd.read_csv(LABELS_PATH)
dfval = pd.read_csv(VAL_LABELS_PATH)

datagen = ImageDataGenerator(rescale=1./255.)

test_datagen = ImageDataGenerator(rescale=1./255.)
start = time.time()
train_generator = datagen.flow_from_dataframe(
    dataframe=dftrain[:70000],
    directory=TRAIN_IMAGES_PATH,
    x_col="file",
    y_col="age",
    batch_size=128,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    classes=ages,
    target_size=(100, 100))

valid_generator = test_datagen.flow_from_dataframe(
    dataframe=dfval,
    directory=VALIDATION_IMAGES_PATH,
    x_col="file",
    y_col="age",
    batch_size=128,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    classes=ages,
    target_size=(100, 100))

test_generator = test_datagen.flow_from_dataframe(
    dataframe=dftrain[70000:],
    directory=TRAIN_IMAGES_PATH,
    x_col="file",
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(100, 100))

model = Sequential()

"""Prvi Conv2d + Relu """
model.add(Conv2D(filters=64, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (100,100,3)))
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
model.add(Dense(9, activation = "softmax"))


# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=7,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
filepath="weights_best.hdf5"
mcp_save = ModelCheckpoint(filepath,save_best_only=True, save_weights_only=True, verbose=1,monitor='val_acc', mode='max')


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=60,
                    callbacks=[learning_rate_reduction, mcp_save]
)

test_generator.reset()
pred=model.predict_generator(test_generator,
    steps=STEP_SIZE_TEST,
    verbose=1)

pred_bool = (pred > 0.5)



predictions = pred_bool.astype(int)
columns=ages
#columns should be the same order of y_col


dump(model, 'fairface_age_cnn.joblib')
end = time.time()

print('Ukupno vreme: ',end - start)
