import argparse
import numpy as np
from tqdm import tqdm

from modeling.build_model import Pose2Seg
from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import cv2
import json 



# save json as
# img = "path.jpg"
# keypoints = [[person1_keypoints],[person2_keypoints],...]

def load_data(image_dir, key_points_json):
    # load image
    img = cv2.imread(image_dir)
    # load keypoints
    return json.load(open(key_points_json))




def get_imgage_segmentation(model,img, image_id,gt_kpts, save_dir=""):
    height, width = img.shape[0:2]
    output = model([img], [gt_kpts])
    # print(np.shape(output[0]))
    n = len(output[0])
    for j in range(len(output[0])):
            # color bitwise anded mask img
        mask =  output[0][j]
        imgr =  cv2.bitwise_and(img, img, mask=mask)
        # imgr = cv2.cvtColor(imgr, )
        # imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
        plt.imsave(save_dir+f'/{image_id}_{j}.png', imgr)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Pose2Seg Testing")
    parser.add_argument(
        "--weights",
        help="path to .pkl model weight",
        type=str,
    )
    parser.add_argument(
        "--img_dir",
        help="path to image directory",
        type=str,
    )
    parser.add_argument(
        "--kp",
        help="path to key points json",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        help="path to save directory",
        type=str,
    )



    args = parser.parse_args()
    
    print('===========> loading model <===========')
    model = Pose2Seg().cuda()
    model.init(args.weights)
            
    print('===========>   testing    <===========')
    # load data

    image_dir = args.img_dir
    key_points_json = args.kp


    data = load_data(image_dir, key_points_json)

    for image_id,dt in data.items():
        img = cv2.imread(image_dir+'/'+dt['img'])
        key_points = np.array(dt['key_points'])
        model.eval()
        # convert to numpy array
        # key_points = np.array(key_points)
        n = len(key_points)
        
        # flatten key point
        # kp = [i for lis in key_points for p in lis for i in p]
        kp  = []
        for lis in key_points:
            for p in lis:
                kp.append(p)
        # print(len(kp))
        gt_kpts = np.array(kp,dtype=np.float32)
        # print(len(gt_kpts))
        print(np.shape(gt_kpts))
        gt_kpts = gt_kpts.reshape(n,17,3)
        # gt_kpts = gt_kpts.transpose(2,1,0)
        print(np.shape(gt_kpts))
        # print(gt_kpts)
        
        
        # get_imgage_segmentation(model,img, image_id,key_points)
        get_imgage_segmentation(model,img, image_id,gt_kpts, save_dir=args.save_dir)


