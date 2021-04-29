# OCR_RaspberryPi_edgetpu


This repo is combining of text detextion and text recognition to be OCR implemented on Raspberry pi 4 with Coral USB accelerator.

We will use the [EAST: An Efficient and Accurate Scene Text Detector](https://github.com/argman/EAST) to be text detector and [Convolutional Recurrent Neural 
Network for End-to-End Text Recognition - TensorFlow 2](https://github.com/FLming/CRNN.tf2) to be text recognition which is a re-implementation of the CRNN network, build by TensorFlow 2, here is [the official repository of CRNN](https://github.com/bgshih/crnn) implemented by [bgshih](https://github.com/bgshih)



[Quantized EAST model](https://tfhub.dev/sayakpaul/lite-model/east-text-detector/int8/2)



