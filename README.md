# CNN
# Convolutional Neutral Network
# Overview:
This project focuses on training a CNN on the Street View House Numbers (SVHN) dataset to classify images of digits. The dataset presents unique challenges due to its real-world nature, including variations in lighting, scale, and rotation of digits. The goal is to build a model that accurately classifies digits despite these challenges.
Such a system can be useful in numerous applications, such as automatic number plate recognition, postal address scanning, or digit classification in mobile applications. The project will help participants implement concepts learned in computer vision by developing a robust CNN model that can handle real-world digit classification.
# Dataset:
The dataset used for this project is the Street View House Numbers (SVHN) dataset, which contains over 600,000 labeled digit images. The dataset is available for download from the following link:
SVHN Dataset

# Introduction
This project focuses on training a Convolutional Neural Network (CNN) to classify images of digits from the Street View House Numbers (SVHN) dataset. The SVHN dataset presents unique challenges due to its real-world nature, including variations in lighting, scale, and rotation of digits. The goal is to build a model that accurately classifies digits despite these challenges.
Dataset:
The SVHN dataset contains over 600,000 labeled digit images. The dataset is available in .mat file format, which is loaded using the scipy.io.loadmat function.
# Preprocessing
1.	Loading Data: The dataset is loaded from .mat files using a helper function.
2.	Reshaping Data: The shape of the input data is adjusted to match the expected format for the CNN.
3.	Label Adjustment: The labels are adjusted so that the digit ‘0’ is correctly represented.
4.	One-Hot Encoding: The labels are one-hot encoded to be used in the categorical cross-entropy loss function.
5.	Normalization: The pixel values of the images are normalized from the range 0-255 to 0-1.
Sample Images
Sample images from the training set are displayed to give an overview of the dataset. These images illustrate the diversity and complexity of the digits in various real-world conditions.
Model Architecture
The CNN model is built using TensorFlow/Keras. The architecture includes:
•	Convolutional Layers: Three convolutional layers with ReLU activation and max pooling.
•	Flatten Layer: To convert the 2D matrix data to a vector.
•	Dense Layers: Two dense layers, including the output layer with softmax activation for multi-class classification.
# Training
The model is compiled with the Adam optimizer and categorical cross-entropy loss function. It is trained for 10 epochs with a batch size of 64. During training, the model’s accuracy and loss are monitored on both the training and validation datasets.
Evaluation
The model’s performance is evaluated on the test set, achieving a test accuracy of approximately 87.65%. This indicates that the model generalizes well to unseen data, effectively handling the variations present in the SVHN dataset.
# Visualization
The training and validation accuracy and loss are plotted over the epochs to visualize the model’s performance. These plots help in understanding how well the model is learning and if there are any signs of overfitting or underfitting.
Conclusion
The CNN model successfully classifies digits from the SVHN dataset with a high degree of accuracy. The project demonstrates the application of convolutional neural networks in handling real-world image classification tasks, highlighting the importance of preprocessing, model architecture, and evaluation metrics.



