# FruityVision - Fruit Identification Using Machine Learning

## Overview

FruityVision is a machine learning-based fruit classification system that utilizes Convolutional Neural Networks (CNNs) to identify various fruits. It is built using the Fruits-360 dataset and provides additional functionality, such as displaying nutritional information for the detected fruits. The project features a user-friendly interface built with Streamlit, enabling real-time fruit recognition via webcam or uploaded images.

## Features

- **Fruit Classification**: Identifies fruit images using a trained CNN model.
- **Real-Time Detection**: Supports image capture from a webcam.
- **User Interface**: Interactive UI using Streamlit for easy accessibility.
- **Nutritional Information**: Displays nutritional details of detected fruits.
- **Efficient Image Processing**: Preprocessing techniques such as resizing, normalization, and augmentation improve model performance.

## Dataset

The project uses the **Fruits-360 dataset**, which consists of:

- **141 unique classes** of fruits, vegetables, and nuts.
- **94,110 images** split into training (70,491 images) and test (23,619 images) sets.
- **Fixed image resolution**: 100x100 pixels in RGB format.

## Model Architecture

The fruit classification model is built using a CNN architecture:

1. **Convolutional and Pooling Layers**: Extracts spatial features from fruit images.
2. **Flatten Layer**: Converts feature maps into a one-dimensional vector.
3. **Fully Connected Layers**: Classifies fruits based on extracted features.
4. **Output Layer**: Uses softmax activation to predict fruit categories.

### Hyperparameters

- **Batch Size**: 32
- **Image Size**: 100x100 pixels
- **Epochs**: 20
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Evaluation Metric**: Accuracy

## Training and Performance

- The model was trained using **data augmentation techniques** to improve generalization.
- Achieved **99.2% validation accuracy** after 20 epochs.
- Final test accuracy: **97.38%** with minimal loss (\~0.2335).

## Installation & Setup

### Prerequisites

- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy, Pandas, Matplotlib
- Streamlit

<!--### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FruityVision.git
   cd FruityVision
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```-->

## Usage

- Open the Streamlit UI.
- Use the webcam or upload an image of a fruit.
- The model classifies the fruit and displays its nutritional information.

## Future Enhancements

- Integration with **YOLOv8/Faster R-CNN** for multi-fruit detection.
- **3D detection** using stereo vision for robotic applications.
- **Edge deployment** using TensorFlow Lite or ONNX.
- **IoT integration** for smart farming and inventory management.
- **Augmented Reality (AR)** features for interactive fruit details.

<!--## Contributors

- **Phanindra Kumar Allada**
- **Sravya Puthi**
- **Abhishek Parsi**
- **Sayee Ajith Ram Reddy Rajula**

## License

This project is licensed under the MIT License. Feel free to use and modify it.-->

## References

- [Fruits-360 Dataset](https://www.kaggle.com/datasets/moltean/fruits)
- [Nutritional Data](https://www.kaggle.com/datasets/yoyoyloloy/fruits-and-vegetables-nutritional-values)
