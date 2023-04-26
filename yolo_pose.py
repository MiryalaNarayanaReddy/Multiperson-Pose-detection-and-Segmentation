import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pyplot as plt
import cv2
import numpy as np
import tqdm
import json
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model

model = load_model()

def run_inference(image):
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (567, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 567, 960])
    if torch.cuda.is_available():
      image = image.half().to(device)
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
      output, _ = model(image)
    return output, image


def draw_keypoints(output, image):
    output = non_max_suppression_kpt(output, 
                                    0.25, # Confidence Threshold
                                    0.65, # IoU Threshold
                                    nc=model.yaml['nc'], # Number of Classes
                                    nkpt=model.yaml['nkpt'], # Number of Keypoints
                                    kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        Keypoints = []
        num_kpts = output.shape[0]
        w = nimg.shape[1]
        h = nimg.shape[0]

        for idx in range(output.shape[0]):
            kpts = output[idx, 7:].T
            # kps =  plot_skeleton_kpts(nimg, kpts,3)
            kp = []

            for i in range(0, len(kpts), 3):
                # if kpts[i + 2] > 0.1:
                    # cv2.circle(nimg, (int(kpts[i]), int(kpts[i + 1])), 3, (0, 255, 0), -1)

                
                kp.append([int(kpts[i])*(h/640), int(kpts[i+1]*(w/640)), float(kpts[i+2])])
            Keypoints.append(kp)
            # Keypoints.append(kps)
    return nimg, Keypoints

# img = cv2.imread('kv.jpg')
# outputs, img = run_inference(img)
# keypoint_img = draw_keypoints(outputs, img)


def pose_estimation_video(filename):
    cap = cv2.VideoCapture(filename)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(number_of_frames)
    # VideoWriter for saving the video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    coco_data = {}    
    # use tqdm to print progress bar
    # os.mkdir("results")
    os.mkdir("results/images")
    1000
    # for i in tqdm.tqdm(range(number_of_frames)):
    for i in tqdm.tqdm(range(10)):
        (ret, frame) = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame1 = cv2.resize(frame, (int(640), int(640)))
            plt.imsave("results/images/"+ str(i).rjust(6,'0')+'.jpg',frame)
            output, frame = run_inference(frame)
            frame,kpts = draw_keypoints(output, frame)

            coco_data[str(i).rjust(6,'0')] = {'img':str(i).rjust(6,'0')+'.jpg','key_points':kpts}
            frame = cv2.resize(frame, (int(cap.get(3)), int(cap.get(4))))
            out.write(frame)
            # cv2.imshow('Pose estimation', frame)
        else:
            break

        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break

    with open('results/output.json', 'w') as outfile:
        json.dump(coco_data, outfile)
    


    cap.release()
    out.release()
    cv2.destroyAllWindows()

pose_estimation_video('kv.mp4')


