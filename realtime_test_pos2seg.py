import argparse
import time
from pathlib import Path
import json
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import random
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import matplotlib.pyplot as plt
import sys

from models.experimental import attempt_load

from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import os

from tqdm import tqdm
from modeling.build_model import Pose2Seg
from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
from pycocotools import mask as maskUtils

def get_imgage_segmentation(model,img, image_id,gt_kpts, save_dir=""):
    height, width = img.shape[0:2]
    output = model([img], [gt_kpts])
    # print(np.shape(output[0]))
    n = len(output[0])
    # create green image 
    green_mask = np.zeros((height, width, 3), np.uint8)
    red_mask = np.zeros((height, width, 3), np.uint8)
    blue_mask = np.zeros((height, width, 3), np.uint8)
    yellow_mask = np.zeros((height, width, 3), np.uint8)
    violet_mask = np.zeros((height, width, 3), np.uint8)
    orange_mask = np.zeros((height, width, 3), np.uint8)
    cyan_mask = np.zeros((height, width, 3), np.uint8)
    magenta_mask = np.zeros((height, width, 3), np.uint8)
    pink_mask = np.zeros((height, width, 3), np.uint8)
    brown_mask = np.zeros((height, width, 3), np.uint8)

    

    green_mask[:] = (0, 255, 0)
    red_mask[:] = (255, 0, 0)
    blue_mask[:] = (0, 0, 255)
    yellow_mask[:] = (0, 255, 255)
    violet_mask[:] = (255, 0, 255)
    orange_mask[:] = (0, 165, 255)
    cyan_mask[:] = (255, 255, 0)
    magenta_mask[:] = (255, 0, 255)
    pink_mask[:] = (255, 192, 203)
    brown_mask[:] = (42, 42, 165)

    color_masks = [green_mask, red_mask, blue_mask, yellow_mask, violet_mask, orange_mask, cyan_mask, magenta_mask, pink_mask, brown_mask]
    
    
    mask = np.zeros((height, width, 3), np.uint8)


    for j in range(len(output[0])):
            # color bitwise anded mask img
        m = cv2.bitwise_and(color_masks[j%10], color_masks[j%10], mask=output[0][j])
        mask = cv2.add(mask, m)
        # imgr =  cv2.bitwise_and(img, img, mask=mask)

        # plt.imsave(save_dir+f'/{image_id}_{j}.png', imgr)
    alpha = 0.5
    imgr =  cv2.addWeighted(img, alpha , mask, 1-alpha, 0)
    # plt.imsave(save_dir+f'/{image_id}.png', imgr)
    return imgr


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
  

def detect(pos2seg_model,save_img=False):
    
    pos2seg_model.eval()

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



    
    source, weights,weights_pos2seg, view_img, save_txt, imgsz, trace,save_segimg = opt.source, opt.weights,opt.weights_pos2seg, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace,opt.save_segimg
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size


    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16


    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    coco_path = str(save_dir) + '/data'
    pos2seg_results = str(save_dir) + '/pos2seg_results'
    os.mkdir(coco_path)
    os.mkdir(pos2seg_results)
    os.mkdir(coco_path+ '/images')

   
    image_id = 0
    coco_data = {}
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        
 
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            img_copy = cv2.cvtColor(im0.copy(), cv2.COLOR_BGR2RGB)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique(): # unique classes detected
                    # print("class name = ",names[int(c)],"id = ",c)
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # write bbox to json file 
                cropped_img = None
                
                # Write results
                # create coco dataset
                image_id += 1
                plt.imsave(coco_path+ f"/images/{str(image_id).rjust(6,'0')}.jpg",cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
                key_points = []

                for *xyxy, conf, cls in reversed(det): # for bounding box person in image

                    # if cls:
                    if cls==0: # if person
                        
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            # cv2.imshow(cropped_img)
                            # .show()
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            # print("log = ,",xywh)

                        if save_img or view_img:  # Add bbox to image
                            
                            image = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]

                            # do pose estimation on this
                            with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.6) as pose:
                                                            
                                results = pose.process(image)
                               
                                # Draw the pose annotation on the image.
                                image.flags.writeable = True
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                # get only coco pose landmarks
                                
                                if results.pose_landmarks == None:
                                    continue

                                coco_points = convert_blaze_to_coco(results.pose_landmarks.landmark)
                                # print("coco points = ",coco_points)
                                for point in coco_points:
                                    cv2.circle(image, (int(point.x * image.shape[1]), int(point.y * image.shape[0])), 4, (0, 0, 255), 1)
                                kps = []
                                for point in coco_points:
                                    kps.append([int(xyxy[0])+int(point.x * image.shape[1]), int(xyxy[1])+int(point.y * image.shape[0]), 2])
                                key_points.append(kps)
                                
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

                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


                            # copy back the cropped array back into big array
                            
                            im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = image

                            # plt.imsave(f'{random.randint(10,100)}.jpg' , im0)

                            # label = f'{names[int(cls)]} {conf:.2f}'
                            # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            imgr = None
            if len(key_points) != 0:
                # save coco data
                coco_data[str(image_id)] = {"img":str(image_id).rjust(6,'0')+'.jpg',"key_points":key_points}
                
                # segmented result
                # convert to numpy array
                # key_points = np.array(key_points)
                n = len(key_points)
                
                # flatten key point
                # kp = [i for lis in key_points for p in lis for i in p]
                kp  = []
                for lis in key_points:
                    for p in lis:
                        for i in p:
                            kp.append(i)
                # print(len(kp))
                gt_kpts = np.array(kp,dtype=np.float32)
                # print(len(gt_kpts))
                # print(np.shape(gt_kpts))
                gt_kpts = gt_kpts.reshape(n,17,3)
                # gt_kpts = gt_kpts.transpose(2,1,0)
                print(np.shape(gt_kpts))
                # print(gt_kpts)
                # get_imgage_segmentation(model,img, image_id,key_points)
                imgr = get_imgage_segmentation(pos2seg_model,img_copy, image_id,gt_kpts,save_dir=pos2seg_results)


            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            # if cropped_img is not None:

            

            # Save results (image with detections)
            im = im0
            if save_segimg and imgr is not None:
                im = imgr
            if save_img:
                if dataset.mode == 'image':
                    # cv2.imwrite(save_path, im0)
                    cv2.imwrite(save_path, im)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            # fps, w, h = 30, im0.shape[1], im0.shape[0]
                            fps, w, h = 30, im.shape[1], im.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    # vid_writer.write(im0)
                    vid_writer.write(im)
                
                
                
                
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

    with open(coco_path+'/coco_data.json', 'w') as fp:
        json.dump(coco_data, fp)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--weights_pos2seg', nargs='+', type=str, default='pos2seg.pkl', help='model.pt path(s)')
    parser.add_argument('--save_segimg', type=int,default=1, help='save segmented image')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    pos2seg_model = Pose2Seg().cuda()
    pos2seg_model.init(opt.weights_pos2seg)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect(pos2seg_model)
