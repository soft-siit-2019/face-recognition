import numpy as np
import cv2
from joblib import load
import dlib
from imutils import face_utils


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

race_model_path = './models/cnn_race_p2.joblib'

# models
model_race = load(race_model_path)

race_array = ['CAUCASIAN', 'BLACK', 'ASIAN', 'INDIAN', 'OTHER', 'OTHER']

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

        test_race = model_race.predict(np.array([img_face]))

        race_num = np.argmax(test_race)

        text += race_array[race_num]

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
