import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


def convert_blaze_to_coco(blazepose_landmarks):
    coco_points = []

    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.NOSE])

    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.LEFT_EYE])
    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.RIGHT_EYE])
    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.LEFT_EAR])
    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.RIGHT_EAR])

    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])

    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])

    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.LEFT_HIP])
    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP])

    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])
    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
    coco_points.append(blazepose_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])


    return coco_points
  
  

# For webcam input:
cap = cv2.VideoCapture('./yolov7/kv.mp4')


def 
with mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.8) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # run with detect.py to get bounding boxes and crop  and then detect pose find copy back




    results = pose.process(image)


    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # get only coco pose landmarks
    


    coco_points = convert_blaze_to_coco(results.pose_landmarks.landmark)
    # print(coco_points)

    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.pose_landmarks,
    #     mp_pose.POSE_CONNECTIONS,
    #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    for point in coco_points:
        cv2.circle(image, (int(point.x * image.shape[1]), int(point.y * image.shape[0])), 4, (0, 0, 255), 1)
    
    # line between eyes

    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y * image.shape[0])),(255,0,0),2)

    # line between eyes and nose
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image.shape[0])),(255,0,0),2)
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image.shape[0])),(255,0,0),2)

    # line between eye and  ear
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y * image.shape[0])),(255,0,0),2)
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y * image.shape[0])),(255,0,0),2)

    # line between shoulder and ear
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y * image.shape[0])),(255,0,0),2)
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y * image.shape[0])),(255,0,0),2)

    # line between shoulder and hip
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image.shape[0])),(255,0,0),2)
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image.shape[0])),(255,0,0),2)

    # line between hip and knee
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image.shape[0])),(255,0,0),2)
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image.shape[0])),(255,0,0),2)

    # line between knee and ankle
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image.shape[0])),(255,0,0),2)
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image.shape[0])),(255,0,0),2)

    # line between shoulder and elbow
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image.shape[0])),(255,0,0),2)
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image.shape[0])),(255,0,0),2)

    # line between elbow and wrist
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image.shape[0])),(255,0,0),2)
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image.shape[0])),(255,0,0),2)

    # line between shoulders
    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0])),(255,0,0),2)

    # line between hips

    cv2.line(image,(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image.shape[0])),(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image.shape[0])),(255,0,0),2)


   
    # print(results.pose_landmarks)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()