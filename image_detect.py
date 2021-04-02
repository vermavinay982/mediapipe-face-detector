import cv2
import mediapipe as mp
# pip install mediapipe

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

face_detection = mp_face.FaceDetection(min_detection_confidence=0.4)

file_path = 'crowd.jpg'
img = cv2.imread(file_path)

# Converting image from BGR to RGB
rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face_detection.process(rgb_image)

if results.detections:
    annotated_img = img.copy()
    for detection in results.detections:
        print(detection)
        mp_draw.draw_detection(annotated_img, detection)

annotated_img = cv2.resize(annotated_img, (800, 400))
cv2.imwrite('crowd_output.jpg', annotated_img)
cv2.imshow('Test', annotated_img)
cv2.waitKey(0)




