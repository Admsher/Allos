# Allos Submision

## Setup
Ensure you have the following packages installed:

Flask: A lightweight web framework for building web applications in Python.


NumPy: A powerful library for numerical computing with support for multi-dimensional arrays and matrices.


Pandas: A versatile library for data manipulation and analysis.


TensorFlow (tf): An open-source machine learning framework for building and training models.


OpenCV (cv2): A library for computer vision and image processing tasks.


To run the proram locally , just locate in the directory and run the command

```bash python app.py

## Model ##
MobileNetV2 is a convolutional neural network architecture specifically designed for efficient deep learning inference on mobile and embedded devices. It serves as an ideal backbone network for object detection tasks, offering a balance between accuracy and computational efficiency, making it suitable for deployment in resource-constrained environments.

Key Features:
Efficient Architecture: MobileNetV2 introduces inverted residual blocks and linear bottlenecks to optimize computational resources, allowing for efficient feature extraction.

Inverted Residuals: Lightweight depthwise convolutions followed by pointwise convolutions with a bottleneck layer enable more efficient use of computational resources.

Linear Bottlenecks: By employing linear activations in bottleneck layers, MobileNetV2 reduces computational cost and improves gradient flow during training.

Width Multiplier and Resolution Multiplier: These hyperparameters offer flexibility to adjust the model's size and computational complexity according to the requirements of the target device or application.


## Deployment 
Used Gunicorn and apache2 on Ubuntu 24.04 to hostthe app on a local server  (http://10.30.30.195:8000/)


Please note that if the site is unresponsive, I might have killed to save battery power and storage of local laptop. So please just send me a mail(f20211826@goa.bits-pilani.ac.in) if the problem is persisting and I will re initiate the server whenever needed.

## Result
A user can expect a huge JSON format array after the process which will contain relevant information about the image used in object detection.
