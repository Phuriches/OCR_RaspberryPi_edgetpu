
# import the necessary packages
import glob
import os
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np
import argparse
import math
import imutils
import time
import cv2
import importlib.util
import urllib.request
import pathlib
import yaml
import decoders
from pathlib import Path
from PIL import Image
from threading import Thread

url = 'http://192.168.0.5:8080/videofeed'

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str,
    default='east_model_int8.tflite', help="path to input TFLite EAST text detector")
ap.add_argument("-w", "--width", type=int, default=320,
    help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
    help="resized image height (should be multiple of 32)")
ap.add_argument("-p", "--padding", type=float, default=0.0,
    help="amount of padding to add to each border of ROI")
ap.add_argument('--edgetpu', action='store_true',
     help='Use Coral Edge TPU Accelerator to speed up detection')
ap.add_argument('--ocr', default="CRNN_float16.tflite",
    help="Path to a binary .pb or .onnx file contains trained recognition network", )
ap.add_argument('--thr', type=float, default=0.5,
    help='Confidence threshold.')
ap.add_argument('--nms', type=float, default=0.4,
    help='Non-maximum suppression threshold.')
ap.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
    default=None)
ap.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
    default=None)
ap.add_argument("-v", "--video", type=str,
    help="path to optinal input video file")
ap.add_argument('--config', type=Path, required=True, 
    help='The config file path.')
args_config = ap.parse_args()
args = vars(ap.parse_args())

with args_config.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['dataset_builder']

class WebcamVideoStream:
    def __init__(self, src):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            
    def read(self):
        # return the frame most recently read
        return self.frame
    
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

def fourPointsTransform(frame, vertices):
    vertices = np.asarray(vertices)
    outputSize = (frame.shape[1], frame.shape[0])
    
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    result = cv2.resize(result,(128, 32))
    return result

def cropImage(image, vertices):
    vertices = np.asarray(vertices)
    x_start = int(vertices[1][0])
    y_start = int(vertices[1][1])
    x_end = int(vertices[3][0])
    y_end = int(vertices[3][1])
    crop_img = image[y_start:y_end , x_start:x_end]
    crop_img = cv2.resize(crop_img,(128, 32))
    return crop_img

def decodeBoundingBoxes(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if (score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

def convert_to_tensor(arr):
  tensor = tf.convert_to_tensor(arr, dtype=tf.uint8)
  return tensor

if args["edgetpu"]:
    mean = np.array([123.68, 116.779, 103.939][::-1], dtype="uint8")
else:
    mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")

mean_tensor = convert_to_tensor(mean)

def preprocess_image(image, mean):
    if args["edgetpu"]:
        image = image.astype("uint8")
    else:
        image = image.astype("float32")
    image -= mean
    return image


def run_ocr_inference(image, model_path):
    # initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image = np.expand_dims(image, 0)
    interpreter.set_tensor(input_details[0]['index'], image)
    
    interpreter.invoke()
    output = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
    #output = convert_to_tensor(output)
    return output

GRAPH_NAME = args["east"]
OCR = args["ocr"]
confThreshold = args["thr"]
nmsThreshold = args["nms"]
IM_NAME = args["image"]
IM_DIR = args["imagedir"]
decoder = decoders.CTCGreedyDecoder(config['table_path'])
padding = args["padding"]

if (IM_NAME and IM_DIR):
    print('Error! Please only use the --image argument or the --imagedir argument, not both. Issue "python TFLite_detection_image.py -h" for help.')
    sys.exit()

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if args["edgetpu"]:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if args["edgetpu"]:
        from tensorflow.lite.python.interpreter import load_delegate

if args["edgetpu"]:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'east_model_int8.tflite'):
        GRAPH_NAME = 'east_model_int8_edgetpu.tflite'
  
# Get path to current working directory
CWD_PATH = os.getcwd()
# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,GRAPH_NAME)
PATH_TO_OCR = os.path.join(CWD_PATH, OCR)

if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*')

elif IM_NAME:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_NAME)
    images = glob.glob(PATH_TO_IMAGES)
    
# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
vs = None
orig = None
orig = None
indices = None
boxes = None
vertices = None
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

if IM_NAME is not None or IM_DIR is not None:
    
    for image_path in images:
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (400,400))    
        orig = frame.copy()
        clear_orig = orig.copy()
        # if our frame dimensions are None, we still need to compute the
        # ratio of old frame dimensions to new frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            rW = W / float(newW)
            rH = H / float(newH)
            
        if args.get("edgetpu", True):
            interpreter = tflite.Interpreter(PATH_TO_CKPT, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        else:
            interpreter = tf.lite.Interpreter(PATH_TO_CKPT)

        input_details = interpreter.get_input_details()
        interpreter.allocate_tensors()
        
        # perform inference and parse the outputs
        frame = cv2.resize(frame, (newW, newH))
        frame = preprocess_image(frame, mean)
        frame = np.expand_dims(frame, 0)
        interpreter.set_tensor(input_details[0]['index'], frame)
        interpreter.invoke()
        
        scores = interpreter.tensor(
            interpreter.get_output_details()[0]['index'])()
        geometry = interpreter.tensor(
            interpreter.get_output_details()[1]['index'])()
        scores = np.transpose(scores, (0, 3, 1, 2)) 
        geometry = np.transpose(geometry, (0, 3, 1, 2))
        [boxes, confidences] = decodeBoundingBoxes(scores, geometry, confThreshold)

        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)

        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            #Adjust bounding box size for model int8
            if args["edgetpu"]:
                vertices[1][0] -= padding
                vertices[1][1] -= padding
                vertices[3][0] += padding
                vertices[3][1] += padding

            start_point = (tuple(vertices[1]))
            end_point = (tuple(vertices[3]))
            
            if args.get("ocr", True):
                interpreter_ocr = tflite.Interpreter(PATH_TO_OCR)
                
            input_details_ocr = interpreter_ocr.get_input_details()
            interpreter_ocr.allocate_tensors()
            
            cropped = fourPointsTransform(clear_orig, vertices)
            cropped = np.float32(cropped)   
            outputs = run_ocr_inference(cropped, OCR)
            
            if not isinstance(outputs, tuple):
               outputs = decoder(outputs)
            text = str(outputs[0].numpy())
            text = text[2:-1]
            cv2.putText(orig, text, (int(vertices[1][0]), int(vertices[1][1])), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.rectangle(orig, start_point, end_point, (0, 0, 255), 2)
               
        cv2.imshow("Text Detection", orig)
        cv2.waitKey(0)
else:
    # if a video path was not supplied, grab the reference to the web cam
    if not args.get("video", False):
        print("Starting video stream...")
        vs = WebcamVideoStream(url).start()
        time.sleep(1.0)

    # otherwise, grab a reference to the video file
    else:
        #vs = cv2.VideoCapture(args["video"])
        print("Got the video")
        vs = WebcamVideoStream(args["video"]).start()
        
    clear_orig = None
    print("Press q to capture")
    while True:
        frame = vs.read()
        frame = cv2.resize(frame, (640,360))
        orig = frame.copy()
        clear_orig = orig.copy()
        # check to see if we have reached the end of the stream
        if frame is None:
            break
        # if our frame dimensions are None, we still need to compute the
        # ratio of old frame dimensions to new frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            rW = W / float(newW)
            rH = H / float(newH)
    
        frame = preprocess_image(orig, mean)
        if args.get("edgetpu", True):
            interpreter = tflite.Interpreter(PATH_TO_CKPT, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        else:
            interpreter = tf.lite.Interpreter(PATH_TO_CKPT)
            
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        model_h = input_details[0]['shape'][1]
        model_w = input_details[0]['shape'][2]
        frame = cv2.resize(frame,(model_w, model_h))
        frame = np.expand_dims(frame, 0)
        
        # perform inference and parse the outputs
        interpreter.set_tensor(input_details[0]['index'], frame)
        interpreter.invoke()
        scores = interpreter.tensor(
            interpreter.get_output_details()[0]['index'])()
        geometry = interpreter.tensor(
            interpreter.get_output_details()[1]['index'])()
        scores = np.transpose(scores, (0, 3, 1, 2)) 
        geometry = np.transpose(geometry, (0, 3, 1, 2))

        [boxes, confidences] = decodeBoundingBoxes(scores, geometry, confThreshold)

        #Apply NMS
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
        
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            
            vertices[1][0] -= padding*3
            vertices[1][1] -= padding*1.5
            vertices[3][0] += padding*6
            vertices[3][1] += padding*1.5
            
            start_point = (tuple(vertices[1]))
            end_point = (tuple(vertices[3]))
            
            cv2.rectangle(orig, start_point, end_point, (0, 0, 255), 2)
    
        cv2.imshow("Text Detection", orig)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vs.stop()
    cv2.destroyAllWindows()
    ###### For OCR ######
    for i in indices:
        # get 4 corners of the rotated rect
        vertices = cv2.boxPoints(boxes[i[0]])
        # scale the bounding box coordinates based on the respective ratios
        for j in range(4):
            vertices[j][0] *= rW
            vertices[j][1] *= rH
        
        vertices[1][0] -= padding*3
        vertices[1][1] -= padding*1.5
        vertices[3][0] += padding*6
        vertices[3][1] += padding*1.5
        
        start_point = (tuple(vertices[1]))
        end_point = (tuple(vertices[3]))
        if args.get("ocr", True):
            interpreter_ocr = tflite.Interpreter(PATH_TO_OCR)
            
        input_details_ocr = interpreter_ocr.get_input_details()
        interpreter_ocr.allocate_tensors()
        cropped = cropImage(clear_orig, vertices)

        cropped = np.float32(cropped)
        outputs = run_ocr_inference(cropped, OCR)

        if not isinstance(outputs, tuple):
           outputs = decoder(outputs)
        text = str(outputs[0].numpy())
        text = text[2:-1]
        cv2.putText(orig, text, (int(vertices[1][0]), int(vertices[1][1])), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (100, 0, 0), 2)
        cv2.rectangle(orig, start_point, end_point, (0, 0, 255), 2)
    
    cv2.imshow("OCR Image", orig)
    cv2.waitKey(0)
    
# close all windows
cv2.destroyAllWindows()
