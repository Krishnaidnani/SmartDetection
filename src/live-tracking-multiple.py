import numpy as np
import cv2
import mediapipe as mp
from tf_keras.models import load_model

categories = np.array(['Distracted', 'Attentive'])
threshold = 0.7
model = load_model('training_data3.h5')


mp_face_detection = mp.solutions.face_detection
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_face_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    return results

def draw_face_rectangle(image, face, label):
    bboxC = face.location_data.relative_bounding_box
    h, w, _ = image.shape
    x_min = int(bboxC.xmin * w)
    y_min = int(bboxC.ymin * h)
    x_max = x_min + int(bboxC.width * w)
    y_max = y_min + int(bboxC.height * h)
    

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    

    cv2.rectangle(image, (x_min, y_min - 30), (x_max, y_min), (0, 255, 0), -1)
    cv2.putText(image, label, (x_min + 5, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def extract_keypoints(holistic_result, face_result):

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in holistic_result.pose_landmarks.landmark]).flatten() if holistic_result.pose_landmarks else np.zeros(33 * 4)


    face = np.array([[res.x, res.y, res.z] for res in holistic_result.face_landmarks.landmark]).flatten() if holistic_result.face_landmarks else np.zeros(468 * 3)


    lh = np.array([[res.x, res.y, res.z] for res in holistic_result.left_hand_landmarks.landmark]).flatten() if holistic_result.left_hand_landmarks else np.zeros(21 * 3)


    rh = np.array([[res.x, res.y, res.z] for res in holistic_result.right_hand_landmarks.landmark]).flatten() if holistic_result.right_hand_landmarks else np.zeros(21 * 3)


    bbox = np.array([face_result.location_data.relative_bounding_box.xmin,
                     face_result.location_data.relative_bounding_box.ymin,
                     face_result.location_data.relative_bounding_box.width,
                     face_result.location_data.relative_bounding_box.height]) if face_result else np.zeros(4)


    keypoints = np.concatenate([pose, face, lh, rh, bbox])


    keypoints = keypoints[-1662:] 

    return keypoints


cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(min_detection_confidence=0.75) as face_detection, \
     mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    
  
    face_sequences = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

       
        face_results = mediapipe_face_detection(frame, face_detection)

        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        holistic_result = holistic.process(image_rgb)

        
        face_labels = {}

        if face_results.detections:
            for idx, face in enumerate(face_results.detections):
               
                keypoints = extract_keypoints(holistic_result, face)
                
                
                if idx not in face_sequences:
                    face_sequences[idx] = []
                face_sequences[idx].append(keypoints)

               
                face_sequences[idx] = face_sequences[idx][-30:]

          
                if len(face_sequences[idx]) == 30:
                    res = model.predict(np.expand_dims(face_sequences[idx], axis=0))[0]
                    predicted_label = categories[np.argmax(res)] if res[np.argmax(res)] >= threshold else "Attentive"
                    face_labels[idx] = predicted_label

               
                draw_face_rectangle(frame, face, face_labels.get(idx, "Uncertain"))

   
        mp_drawing.draw_landmarks(frame, holistic_result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

   
        cv2.imshow('Face Detection with Holistic', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
