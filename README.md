# [OCR based on Deep Learning implemented on Raspberry Pi4 with Coral USB Accelerator.](https://medium.com/p/fb6f6a933850/edit)

## Introduction
This project will be demonstrated an OCR(Optical Character Recognition) Technology based on Deep Learning by using Raspberry Pi as a Microcontroller, in order to improve its performance working together with Coral USB Accelerator is an interesting choice. It adds an Edge TPU coprocessor to your system, and enabling high-speed inferencing with low consumption and also supports TensorFlow Lite which is a lightweight model for mobile deployment. Completely, this project can recognize in real-time and convert virtually any kind of image or stream video containing typed and printed text into machine-readable text data. It is very easy to use, just open your mobile camera and got the text data!

Part1: Text Detection, there are alternatives pre-trained models can detect text on image or video but we focus on text detectors that can be use for microcontroller deployment such as EAST model, CRAFT model, TextBoxes++, and PaddleOCR. CRAFT and EAST, Their performances are almost the same, but still different. East was chosen for working in this project following its better performance in [A battle result article](https://sayak.dev/optimizing-text-detectors/).

[EAST (Efficient accurate scene text detector)](https://arxiv.org/abs/1704.03155v2), it is a very robust deep learning text detection in natural scenes. The pipeline directly predicts words or text lines of arbitrary orientations and quadrilateral shapes in full images, eliminating unnecessary intermediate steps (e.g., candidate aggregation and word partitioning), with a single neural network. So, It is great option to be our text detector.

Part2 : Text Recognition, which is among the most important and challenging tasks in image-based sequence recognition. Unlike general object recognition, recognizing such sequence-like objects often requires the system to predict a series of object labels, instead of a single label. There are a lot of great work in these categories such as tesseract, keras-ocr, and CRNN. In this mini-project, [CRNN model](https://github.com/FLming/CRNN.tf2) was chosen to implement in recognition part.

[CRNN (Convolutional Recurrent Neural Networks)](https://arxiv.org/abs/1507.05717), it is a implementation of a Deep Neural Network for scene text recognition. The model consists of a CNN stage extracting features which are fed to an RNN stage (Bi-LSTM) and a CTC loss for image-based sequence recognition tasks, such as scene text recognition and OCR. For details, please take a look into their paper.

## Implementation

For image implementation, put the code down below.
```
python -Xfaulthandler OCR_Rpi.py --config mjsynth.yml --east east_model_float16.tflite --image xxx.jpg
```

For video implementation, put the code down below.

```
python -Xfaulthandler OCR_Rpi.py --config mjsynth.yml --edgetpu --padding 8 --nms 0.05 --thr 200
```


