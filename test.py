from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd
from tqdm import trange
from statistics import mean 
from collections import namedtuple
import time

GROUNDTRUTH_DIR = "groundtruths"
DETECTIONS_DIR = "detections"

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval-pytorch', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true',
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

groundtruths_path = os.path.join(args.save_folder, GROUNDTRUTH_DIR)
if not os.path.exists(groundtruths_path):
    os.mkdir(groundtruths_path)

detections_path = os.path.join(args.save_folder, DETECTIONS_DIR)
if not os.path.exists(detections_path):
    os.mkdir(detections_path)

Detection = namedtuple("Detection", "c s x y r b")
# where c is the class, s is the confidence score, 
# (x,y) are the coordinates of the top-left corner of the bounding box 
# and (r,b) the coordinates of the bottom-right corner of the bounding box


def test_net(save_folder, net, cuda, testset, transform, thresh):
    num_images = len(testset)

    inference_time_list = []

    # do the detection on every image
    for idx in trange(num_images):
        try:
            # Get next test image and label
            img = testset.pull_image(idx)
            img_id, annotation = testset.pull_anno(idx)

            #Pre-process the image
            x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
            x = Variable(x.unsqueeze(0))
            if cuda:
                x = x.cuda()

        except:
            print(f"[ ERROR ] Problem loading the image '{img_id}' -> This image will not be included in the test.")
            continue

        # Save the groundtruth information
        groundtruth_file = os.path.join(groundtruths_path, img_id + ".txt")
        with open(groundtruth_file, mode='x') as f:
            for box in annotation:
                label_name = labelmap[box[4]]
                f.write(f"{label_name} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n")

        # forward pass
        start_time = time.time()
        y = net(x)

        inference_time_list.append(time.time() - start_time)

        # Post-process output
        outputs = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        
        detections = []
        # Loop over all the classes
        for idx in range(outputs.size(1)): 
            for jdx in range(outputs.size(2)):
                score = outputs[0, idx, jdx, 0].item()
                if score >= thresh:
                    class_idx = idx
                    bbox = (outputs[0, idx, jdx, 1:]*scale).cpu().numpy()
                    detections.append(Detection(class_idx, score, bbox[0], bbox[1], bbox[2], bbox[3]))

        # Save the detections information
        detections_file = os.path.join(detections_path, img_id + ".txt")
        with open(detections_file, mode='x') as f:
            for detection in detections:
                class_idx, score, x, y, r, b = detection
                label_name = labelmap[class_idx-1]
                f.write(f"{label_name} {round(score, 6)} {int(x)} {int(y)} {int(r)} {int(b)}\n")
        
    print(f"Average Inference Time (s): {mean(inference_time_list)}")


        # pred_num = 0
        # for i in range(detections.size(1)):
        #     j = 0
        #     while detections[0, i, j, 0] >= 0.6:
        #         if pred_num == 0:
        #             with open(filename, mode='a') as f:
        #                 f.write('PREDICTIONS: '+'\n')
        #         score = detections[0, i, j, 0]
        #         label_name = labelmap[i-1]
        #         pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
        #         coords = (pt[0], pt[1], pt[2], pt[3])
        #         pred_num += 1
        #         with open(filename, mode='a') as f:
        #             f.write(str(pred_num)+' label: '+label_name+' score: ' +
        #                     str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
        #         j += 1


def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    device = "cuda" if args.cuda else "cpu"
    net.load_state_dict(torch.load(args.trained_model, map_location=torch.device(device)))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
