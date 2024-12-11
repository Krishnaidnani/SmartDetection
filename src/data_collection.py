import numpy as np
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import os

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


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


Input_data_path = os.path.join("inputData3")
categories = np.array(['Distracted','Attentive'])
sequences = 100
sequence_length = 30

for category in categories:
    for sequence in range(sequences):
        try:
            os.makedirs(os.path.join(Input_data_path,category,str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence = 0.5 ,min_tracking_confidence = 0.5) as holistic:
    for category in categories:

        for sequence in range(sequences):

            for frame_num in range(sequence_length+1):
                    ret,frame = cap.read()
                    image,result = mediapipe_detection(frame,holistic)
                    draw_color_landmarks(image,result)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(category, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(category, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                        keypoints = extract_keypoints(result)
                        data_collection_path = os.path.join(Input_data_path,category,str(sequence),str(frame_num))
                        np.save(data_collection_path,keypoints)


    cap.release()
    cv2.destroyAllWindows()
draw_color_landmarks(frame,result)
plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
plt.title("last captured frame")
plt.show()


