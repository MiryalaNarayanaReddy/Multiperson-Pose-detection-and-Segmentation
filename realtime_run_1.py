import argparse
import numpy as np
from tqdm import tqdm

from modeling.build_model import Pose2Seg
from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import cv2


def test(model, dataset='cocoVal', logger=print):    
    if dataset == 'OCHumanVal':
        ImageRoot = './data/OCHuman/images'
        AnnoFile = './data/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json'
    elif dataset == 'OCHumanTest':
        ImageRoot = './data/OCHuman/images'
        AnnoFile = './data/OCHuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json'
    elif dataset == 'cocoVal':
        ImageRoot = './data/coco2017/val2017'
        AnnoFile = './data/coco2017/annotations/person_keypoints_val2017_pose2seg.json'

    elif dataset == 'image_dir':
        ImageRoot = './data/images'
        AnnoFile = './data/annotations/pose2seg.json'


    datainfos = CocoDatasetInfo(ImageRoot, AnnoFile, onlyperson=True, loadimg=True)
    
    model.eval()
    
    results_segm = []
    imgIds = []
    # have to remove seg mask arg
    # add coco file and check
    # check if it works on images
    # check if it works on video
    # do it in one image
    
    for i in tqdm(range(len(datainfos))):
        rawdata = datainfos[i]
        img = rawdata['data']
        image_id = rawdata['id']
        
        height, width = img.shape[0:2]
        gt_kpts = np.float32(rawdata['gt_keypoints']).transpose(0, 2, 1) # (N, 17, 3)
        print(type(gt_kpts[0]))
        print(gt_kpts.shape)
        gt_segms = rawdata['segms']
        gt_masks = np.array([annToMask(segm, height, width) for segm in gt_segms])
            
        output = model([img], [gt_kpts], [gt_masks])
        print(np.shape(output[0]))
        # model._visualizeOutput(output)
        # mask = output[0][0]
        n = len(output[0])
        for j in range(len(output[0])):
             # color bitwise anded mask img
            mask =  output[0][j]
            imgr =  cv2.bitwise_and(img, img, mask=mask)
            # imgr = cv2.cvtColor(imgr, )
            # imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
            plt.imsave(f'{image_id}_{j}.png', imgr)
           
        # plt.imsave(f'{image_id}.png', final_img)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Pose2Seg Testing")
    parser.add_argument(
        "--weights",
        help="path to .pkl model weight",
        type=str,
    )
    parser.add_argument(
        "--image_dir",
        help="path to image directory",
        type=str,
    )
    parser.add_argument(
        "--video",
        help="path to video",
        type=str,
    )
    parser.add_argument(
        "--OCHuman",
        help="test on OCHuman dataset",
        action='store_true',
    )

    parser.add_argument(
        "--coco",
        help="test on coco dataset",
        action='store_true',
    )
    
    args = parser.parse_args()
    
    print('===========> loading model <===========')
    model = Pose2Seg().cuda()
    model.init(args.weights)
            
    print('===========>   testing    <===========')
    if args.coco:
        test(model, dataset='cocoVal') 
    if args.OCHuman:
        # test(model, dataset='OCHumanVal')
        test(model, dataset='OCHumanTest') 
    if args.video:
        test(model, video=args.video)
    if args.image_dir:
        test(model, image_dir=args.image_dir)


