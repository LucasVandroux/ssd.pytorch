from __future__ import print_function
import sys
import os
import argparse
import time
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
from collections import namedtuple
from tqdm import trange
from statistics import mean 

GROUNDTRUTH_DIR = "groundtruths"
DETECTIONS_DIR = "detections"

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--model', type=str, help='Path to .xml file to load.')
parser.add_argument('--save_folder', default='eval-openvino/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--gpu', action='store_true',
                    help='Use gpu to test model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

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

class SDDOpenVINO:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model_xml_path, device="CPU"):
        """ Load the model 

            Args:
                model_xml_path (str): path to the .xml files representing the model. It is assumed the .bin file with the weights has the same name.
                device (str: 'CPU'): name of the device where to load the model
        """
        # Create path to both files of the model
        model_xml = model_xml_path
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        # Initialize the plugin
        self.plugin = IECore()

        # Read IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        # Check for supported layers
        supported_layers = self.plugin.query_network(network=self.network, device_name=device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            sys.exit(f"Unsupported layers found: {unsupported_layers}.")

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer and output layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

    def get_input_shape(self):
        """ Gets the input shape of the network
        """
        return self.network.inputs[self.input_blob].shape

    def async_inference(self, image):
        """ Makes an asynchronous inference request, given an input image.

        Args:
            image (np.array): numpy array representing the image
        """
        self.exec_network.start_async(request_id=0, 
            inputs={self.input_blob: image})
    
    def sync_inference(self, image):
        """ Return the output of inference

        Args:
            image (np.array): numpy array representing the image

        Returns:
            output (np.array): output of the model
            inference_time (float): time in seconds needed to perform the inference
        """
        start_time = time.time()
        output_ = self.exec_network.infer({self.input_blob: image})

        return output_[self.output_blob], time.time() - start_time

    def wait(self):
        """ Checks the status of the inference request.
        """
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        """ Returns a list of the results for the output layer of the network.
        """
        return self.exec_network.requests[0].outputs[self.output_blob]

    def postprocess_output(self, network_output, width, height, prob_threshold = 0.5):
        """ Post-process the output of the network

        Args:
            network_output: direct output of the network after inference
            width (int): width of the input frame
            height (int): height of the input frame
            prob_threshold (float: 0.5): probability threshold for detections filtering

        Returns:
            list_detections (Detections): list of bounding boxes of the 
                detected objects.
        """
        list_detections = []
        # The net outputs a blob with shape: [1, 1, N, 7], where N is the number of detected 
        # bounding boxes. For each detection, the description has the format: 
        # [image_id, label, conf, x_min, y_min, x_max, y_max]
        bboxes = np.reshape(network_output, (-1, 7)).tolist()

        for bbox in bboxes:
            conf = bbox[2]
            object_class = int(bbox[1])
            if conf >= prob_threshold:
                xmin = int(bbox[3] * width)
                ymin = int(bbox[4] * height)
                xmax = int(bbox[5] * width)
                ymax = int(bbox[6] * height)

                list_detections.append(Detection(object_class, conf, xmin, ymin, xmax, ymax))

        return list_detections


def test_net(save_folder, net, cuda, testset, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test_openvino.txt' #TODO
    num_images = len(testset)
    net_input_shape = net.get_input_shape()

    inference_time_list = []

    # do the detection on every image
    for idx in trange(num_images):
        try:
            # Get next test image and label
            img = testset.pull_image(idx)
            img_id, annotation = testset.pull_anno(idx)

            #Pre-process the image
            p_img = cv2.resize(img, (net_input_shape[3], net_input_shape[2]))
            p_img = p_img.transpose((2,0,1))
            p_img = p_img.reshape(1, *p_img.shape)

        except:
            print(f"[ ERROR ] Problem loading the image '{img_id}' -> This image will not be included in the test.")
            continue
            
        # Save the groundtruth information
        groundtruth_file = os.path.join(groundtruths_path, img_id + ".txt")
        with open(groundtruth_file, mode='x') as f:
            for box in annotation:
                label_name = labelmap[box[4]]
                f.write(f"{label_name} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n")

        output, inference_time = net.sync_inference(p_img)      # forward pass
        detections = net.postprocess_output(output,img.shape[1], img.shape[0], thresh)
        
        inference_time_list.append(inference_time)

        # Save the detections information
        detections_file = os.path.join(detections_path, img_id + ".txt")
        with open(detections_file, mode='x') as f:
            for detection in detections:
                class_idx, score, x, y, r, b = detection
                label_name = labelmap[class_idx-1]
                f.write(f"{label_name} {round(score, 6)} {int(x)} {int(y)} {int(r)} {int(b)}\n")
        
    
    print(f"Average Inference Time (s): {mean(inference_time_list)}")
                

def test_voc():
    # Initialise the class
    net = SDDOpenVINO()
    # Set Probability threshold for detections
    prob_threshold = args.visual_threshold
    #Load the model through `net`
    device = "GPU" if args.gpu else "CPU"
    net.load_model(args.model, device = device)
    print('Finished loading model!')

    # load data
    testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())

    # evaluation
    test_net(args.save_folder, net, args.gpu, testset,
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
