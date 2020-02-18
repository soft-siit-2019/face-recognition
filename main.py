import numpy as np
import cv2
from joblib import load
import dlib
from imutils import face_utils


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

race_model_path = './models/cnn_race_p2.joblib'
ages_model_path = './models/fairface_age_cnn.joblib'
gender_model_path = './models/cnn_gender_p2.joblib'

# models
model_race = load(race_model_path)
model_ages = load(ages_model_path)
model_gender = load(gender_model_path)

race_array = ['CAUCASIAN', 'BLACK', 'ASIAN', 'INDIAN', 'OTHER', 'OTHER']
# ages_array = ['1-5', '6-10', '11-15', '16-19', '20-25', '26-32', '33-40', '41-50', '51-60', '61-70', '71-80', '81-90', '90+']
ages_array = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
gender_array = ['MALE', 'FEMALE']

video = cv2.VideoCapture(0)

while True:
    # Create a frame object
    check, frame = video.read()
    text = ''

    dets = detector(frame, 1)
    num_faces = len(dets)

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(predictor(frame, detection))

    for(i, rect) in enumerate(dets):

        (x, y, w, h) = face_utils.rect_to_bb(rect)

        if x < 0 or y < 0 or w < 64 or h < 64:
            continue

        img_face = frame[y:y + h, x:x + w]
        img_face = cv2.resize(img_face, (64, 64))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_face = img_face / 255.0

        # ages
        img_face2 = dlib.get_face_chip(frame, faces[i])
        img_face2 = cv2.resize(img_face2, (100, 100))
        img_face2 = img_face2 / 255.0

        test_race = model_race.predict(np.array([img_face]))
        test_ages = model_ages.predict(np.array([img_face2]))
        test_gender = model_gender.predict(np.array([img_face]))

        race_num = np.argmax(test_race)
        ages_num = np.argmax(test_ages)
        gender_num = np.argmax(test_gender)

        text += race_array[race_num]

        text += (', ' + ages_array[ages_num])

        text += (', ' + gender_array[gender_num])


        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(text)
        text = ''

    cv2.imshow("Face App", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break


# Shutdown camera
video.release()

cv2.destroyAllWindows()
