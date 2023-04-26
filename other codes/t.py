import cv2
import mediapipe as mp
import numpy as np

mediapipe_to_coco = {
    11: 5,  # left shoulder
    13: 6,  # left elbow
    15: 7,  # left wrist
    12: 2,  # right shoulder
    14: 3,  # right elbow
    16: 4,  # right wrist
    23: 11, # left hip
    25: 12, # left knee
    27: 13, # left ankle
    24: 8,  # right hip
    26: 9,  # right knee
    28: 10, # right ankle
    0: 14,  # nose
    1: 15,  # left eye
    2: 16,  # right eye
    3: 17,  # left ear
    4: -1,  # right ear (not present in COCO)
    8: -1,  # left heel (not present in COCO)
    12: -1, # right heel (not present in COCO)
}

# initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# initialize webcam stream
cap = cv2.VideoCapture(0)

while True:
    # read frame from webcam
    ret, image = cap.read()

    # convert image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect pose landmarks using Mediapipe Pose model
    results = pose.process(image)
    indx = 0
    if results.pose_landmarks:
        # extract 17 COCO keypoints from detected landmarks
        coco_pose = np.zeros((17, 3))
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if i in [11, 12, 23, 24, 0, 1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16]:
                coco_pose[indx] = [landmark.x, landmark.y, landmark.z]
                indx += 1
        # draw keypoints on the image
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for i, point in enumerate(coco_pose):
            if all(point != [0, 0, 0]):
                x, y = int(point[0] * image.shape[1]), int(point[1] * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # display the resulting image
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(1) == ord('q'):
        break

# release resources
cap.release()
cv2.destroyAllWindows()
