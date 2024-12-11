import numpy as np
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from tf_keras.models import load_model

sequence = []
predicted_label = "Uncertain" 
threshold = 0.8
categories = np.array(['Distracted','Attentive'])

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
model = load_model('training_data3.h5')


def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,result

def draw_color_landmarks(image,result):
    mp_drawing.draw_landmarks(image,result.face_landmarks,mp_holistic.FACEMESH_CONTOURS,
                            mp_drawing.DrawingSpec(color=(80,105,15),thickness=1,circle_radius=1),
                            mp_drawing.DrawingSpec(color=(80,250,115),thickness=1,circle_radius=1))
    mp_drawing.draw_landmarks(image,result.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,20,10),thickness=1,circle_radius=1),
                               mp_drawing.DrawingSpec(color=(80,45,120),thickness=1,circle_radius=1))
    mp_drawing.draw_landmarks(image,result.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(115,22,72),thickness=1,circle_radius=1),
                               mp_drawing.DrawingSpec(color=(115,44,230),thickness=1,circle_radius=1))
    mp_drawing.draw_landmarks(image,result.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(235,112,65),thickness=1,circle_radius=1),
                               mp_drawing.DrawingSpec(color=(235,60,210),thickness=1,circle_radius=1))
    
def extract_keypoints(result):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

cap = cv2.VideoCapture(0)   
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret,frame = cap.read()

        image,result = mediapipe_detection(frame, holistic)
        draw_color_landmarks(image,result)

        keypoints = extract_keypoints(result)

        sequence.append(keypoints)
        sequence = sequence[-30:]

        if(len(sequence)==30):
            res = model.predict(np.expand_dims(sequence,axis=0))[0]
            if res[np.argmax(res)] >= threshold:
                predicted_label = (categories[np.argmax(res)])

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)  
        cv2.putText(image, predicted_label, (3, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  
        cv2.imshow('OpenCV ouput', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
