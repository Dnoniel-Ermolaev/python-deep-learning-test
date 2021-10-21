"""
Classification sample

Command line to run:
python ie_classification_sample.py -i image.jpg \
    -m squeezenet1.1.xml -w squeezenet1.1.bin -c imagenet_synset_words.txt
"""

import os
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.4.689\\deployment_tools\\ngraph\\lib")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.4.689\\deployment_tools\\inference_engine\\external\\tbb\\bin")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.4.689\\deployment_tools\\inference_engine\\bin\\intel64\\Release")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.4.689\\deployment_tools\\inference_engine\\external\\hddl\\bin")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.4.689\\opencv\\bin")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.4.689\\python\\python3.9\\openvino\\libs")
os.add_dll_directory("C:\\Users\\ermol\\openvino-virtual-environments\\openvinoenv\\Lib\\site-packages\\pytest")
import warnings
import os
import pytest
import cv2
import sys
import argparse
import numpy as np
import logging as log
from openvino.inference_engine import IECore
sys.path.append('../lib/')


def Sum(A,B):
    sum = A+B
    return sum

def test_Sum():
    assert (5+4) == Sum(5,4)
    
class InferenceEngineClassifier:
    
    def __init__(self, configPath = None, weightsPath = None,
    device = 'CPU', classesPath = None):
        self.ie = IECore()
        self.net = self.ie.read_network(model=configPath)
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)
#        with open(args.labels, 'r') as f:
#           labels = [line.split(',')[0].strip() for line in f]
        return

    def test_get_top(self, prob, topN = 1):

        result =[]
        result = np.squeeze(prob)
    # Get an array of args.number_top class IDs in descending order of probability
        result = np.argsort(result)[-topN :][::-1]

        return result
    
    def test_prepare_image(self, image, h, w):

        #image = image.transpose((2, 0, 1)) 
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
       
        return image

    def classify(self, image):
        probabilites=None
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))

        n, c, h, w = self.net.inputs[input_blob].shape
       
        image = self.test_prepare_image(image, h, w)
        
        output = self.exec_net.infer(inputs = {input_blob: image})
        output = output[out_blob]
        
        return output


def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', help='Path to an .xml file with a trained model.', default='C:\\Users\\ermol\\public\\squeezenet1.1\\FP16\\squeezenet1.1.xml', type=str)
    parser.add_argument('-w', '--weights', help='Path to an .bin file with a trained weights.', default='C:\\Users\\ermol\\public\\squeezenet1.1\\FP16\\squeezenet1.1.bin', type=str)
    parser.add_argument('-i', '--input', help='Path to image file', default='C:\\Users\\ermol\\PycharmProjects\\practice\\monkey.jpeg', type=str)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    parser.add_argument('-c', '--classes', help='File containing classes \
        names', type=str, default=None)
    return parser

#image with doge_class.py

@pytest.mark.parametrize("IMAGE, RESULT",[(str('monkey.jpeg'),373),
(str('wolf_spider.jpg'),77),
(str('cut.jpg'),285)
])
def test_main(IMAGE,RESULT):

    warnings.filterwarnings("ignore")

    model = str('C:\\Users\\ermol\\public\\squeezenet1.1\\FP16\\squeezenet1.1.xml')
    weights = str('C:\\Users\\ermol\\public\\squeezenet1.1\\FP16\\squeezenet1.1.bin')
    input = IMAGE
    device = str('CPU')

    log.info("Start IE classification sample")
    ie_classifier = InferenceEngineClassifier(configPath=model,weightsPath=weights, device=device ,classesPath=None)
    img = cv2.imread(input)
    prob = ie_classifier.classify(img)

    predictions = ie_classifier.test_get_top(prob, 5)

    assert predictions[0] == RESULT
    