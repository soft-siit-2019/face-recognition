import numpy as np
import os
import cv2
import collections
from joblib import dump
import dlib
from imutils import face_utils
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


def load_image(path):
    img = cv2.imread(path)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gs

# Angry, Afraid, Disgusted, Happy, Neutral, Sad, Surprised
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


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx * ny))


KDEF_PATH = '../data' + os.path.sep + 'KDEF_and_AKDEF' + os.path.sep + 'KDEF' + os.path.sep
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')

image_paths = list_files(KDEF_PATH)
print(len(image_paths))

imgs = []
emotions = []
cnt = 0
faces = []

if len(faces) == 0:
    for image_path in image_paths:
        image_directory, image_name = os.path.split(image_path)

        for key in emotion:
            if emotion[key] in image_name:
                img = load_image(image_path)
                cnt += 1
                print(cnt)
                img = cv2.resize(img, (800,800))
                rects = detector(img, 1)
                for (i, rect) in enumerate(rects):
                    (x, y, w, h) = face_utils.rect_to_bb(rect)
                    shape = predictor(img, rect)
                    shape = shape_to_np(shape)
                    faces.append(shape)
                    emotions.append(key)

np.save('imgs_128x128_emotion_labels', emotions)
np.save('faces_emotion', faces)

emotions = np.array(emotions)
faces = np.array(faces)

counter = collections.Counter(emotions)
print(counter)

# Normalize the data
faces = faces / 255.0

X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.15, random_state=42)

X_train = reshape_data(X_train)
X_test = reshape_data(X_test)

print(X_train.shape)
print(X_train)

svc_params = {'C': [0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)
clf_svm = grid_svc.best_estimator_

y_train_pred = clf_svm.predict(X_train)
y_test_pred = clf_svm.predict(X_test)
print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

dump(clf_svm, '../models/emotion_svm.joblib')