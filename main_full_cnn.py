import numpy as np
import cv2
from joblib import load
import dlib
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# paths
race_model_path = './models/cnn_race_p2.joblib'
ages_model_path = './models/cnn_age_p2.joblib'
gender_model_path = './models/cnn_gender_p2.joblib'
emotion_model_path = './models/cnn_emotion.joblib'

# models
model_race = load(race_model_path)
model_ages = load(ages_model_path)
model_gender = load(gender_model_path)
model_emotion = load(emotion_model_path)

race_array = ['CAUCASIAN', 'BLACK', 'ASIAN', 'INDIAN', 'OTHER', 'OTHER']
ages_array = ['1-5', '6-10', '11-15', '16-19', '20-25', '26-32', '33-40', '41-50', '51-60', '61-70', '71-80', '81-90', '90+']
gender_array = ['MALE', 'FEMALE']
emotion_array = ['Angry', 'Afraid', 'Disgusted', 'Happy', 'Neutral', 'Sad', 'Surprised']


video = cv2.VideoCapture(0)

while True:
    # Create a frame object
    check, frame = video.read()
    text = ''

    dets = detector(frame, 1)
    num_faces = len(dets)

    for(i, rect) in enumerate(dets):

        (x, y, w, h) = face_utils.rect_to_bb(rect)

        if x < 0 or y < 0 or w < 64 or h < 64:
            continue

        img_face = frame[y:y + h, x:x + w]
        img_face = cv2.resize(img_face, (64, 64))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_face = img_face / 255.0

        img_face2 = frame[y:y + h, x:x + w]
        img_face2 = cv2.resize(img_face2, (128, 128))
        img_face2 = img_face2 / 255.0

        test_race = model_race.predict(np.array([img_face]))
        test_ages = model_ages.predict(np.array([img_face]))
        test_gender = model_gender.predict(np.array([img_face]))
        test_emo = model_emotion.predict(np.array([img_face2]))

        race_num = np.argmax(test_race)
        ages_num = np.argmax(test_ages)
        gender_num = np.argmax(test_gender)
        emotion_num = np.argmax(test_emo)


        text += race_array[race_num]

        text += (', ' + ages_array[ages_num])

        text += (', ' + gender_array[gender_num])

        text += (', ' + emotion_array[emotion_num])

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
