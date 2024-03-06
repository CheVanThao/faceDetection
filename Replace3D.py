import cv2
import dlib
import numpy as np

# Load the face detector and facial landmarks predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the 2D avatar image (replace with your own 2D avatar image)
avatar = cv2.imread("download.jpeg", cv2.IMREAD_UNCHANGED)

def apply_3d_avatar(frame, face_landmarks):
    # Extract the region of interest (ROI) around the face
    (x, y, w, h) = cv2.boundingRect(np.array(face_landmarks))
    face_roi = frame[y:y+h, x:x+w]
    print(face_roi)

    # Resize the avatar image to match the size of the face ROI
    avatar_resized = cv2.resize(avatar, (w, h))

    # Extract the alpha channel from the avatar image (if available) 
    avatar_resized = cv2.cvtColor(avatar_resized, cv2.COLOR_RGB2RGBA).copy()
    avatar_alpha = avatar_resized[:, :, 3]

    # Apply the avatar as a texture to the face ROI
    for c in range(0, 3):
        face_roi[:, :, c] = face_roi[:, :, c] * (avatar_alpha / 255.0) + avatar_resized[:, :, c] * (avatar_alpha / 255.0)
        # print(x, y, w, h)

    # Replace the face in the frame with the avatar-textured face
    frame[y:y+h, x:x+w] = face_roi

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

        # Apply a 3D avatar effect to the detected face
        frame = apply_3d_avatar(frame, landmarks)

    # Display the frame
    cv2.imshow("Face with 3D Avatar", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
