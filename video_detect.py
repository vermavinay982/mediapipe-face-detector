import cv2
import mediapipe as mp
# pip install mediapipe

class FaceDetect:
    def __init__(self, threshold=0.5):
        self.mp_face = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face.FaceDetection(min_detection_confidence=threshold)
    
    def detect(self, frame):
        # Converting image from BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        detections = results.detections
        frame = self.draw(frame, detections)
        return detections, frame

    def draw(self, frame, detections):
        # frame = cv2.resize(frame, (800, 600))
        if detections:
            for detection in detections:
                self.mp_draw.draw_detection(frame, detection)
        return frame

if __name__=='__main__':
    detector = FaceDetect(threshold=0.7)

    file_path = 'crowd_video.mp4'
    cap = cv2.VideoCapture(file_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, annotated_img = detector.detect(frame)
        annotated_img = cv2.resize(annotated_img, (800, 600))
        cv2.imshow('Test', annotated_img)
        
        if ord('q')==cv2.waitKey(1):
            break

    cap.release()
    cv2.destroyAllWindows()