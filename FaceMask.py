import cv2
import dlib
import numpy as np

# Load the face detector and facial landmarks predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/user/workspace/tensorflow/shape_predictor_68_face_landmarks.dat")

def apply_cartoon_filter(frame, face_landmarks):
    # Extract the region of interest (ROI) around the face
    (x, y, w, h) = cv2.boundingRect(np.array(face_landmarks))
    face_roi = frame[y:y+h, x:x+w]

    # Apply a cartoon filter to the face ROI
    cartoon_filter = cv2.stylization(face_roi, sigma_s=150, sigma_r=0.25)

    # Replace the face in the frame with the cartoon-filtered face
    frame[y:y+h, x:x+w] = cartoon_filter

    return frame

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)
        landmarks = [(p.x, p.y) for p in landmarks.parts()]

        # Apply a cartoon filter to the detected face
        frame = apply_cartoon_filter(frame, landmarks)

    # Display the frame
    cv2.imshow("Face with Cartoon Filter", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
