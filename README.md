# Deep Learning Image Classification Project

This project implements and compares three different deep learning architectures for image classification: **Simple CNN**, **CNN-LSTM**, and **Vision Transformer (ViT)**. It includes Jupyter notebooks for training the models and a Streamlit web application for interactive inference.

## 📂 Project Structure

- **`app.py`**: The main Streamlit application for running inference on images using the trained models.
- **`models.py`**: Contains the PyTorch model definitions (`SimpleCNN`, `CNNLSTM`) and helper functions (`get_vit_model`).
- **`CNN_and_CNN_LSTM.ipynb`**: Jupyter notebook used for training and evaluating the Simple CNN and CNN-LSTM models.
- **`vit-freeze.ipynb`**: Jupyter notebook used for training and evaluating the Vision Transformer (ViT) model (using transfer learning).
- **`requirements.txt`**: List of Python dependencies required to run the project.
- **Model Weights** (files expected in the root directory):
  - `model.pth`: Trained weights for the Simple CNN model.
  - `best_cnn_lstm_model.pth`: Trained weights for the CNN-LSTM model.
  - `natural_images_vit.pth`: Trained weights for the ViT model.

## 🚀 Installation

1.  **Clone or Download the Repository** to your local machine.

2.  **Create a Virtual Environment** (Recommended):

    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    _Note: If you have a GPU, ensure you install the correct version of PyTorch with CUDA support from [pytorch.org](https://pytorch.org/)._

## 🎮 How to Use

### Running the Web App

The project includes a user-friendly web interface built with Streamlit.

1.  Ensure you have the trained model files (`.pth`) in the project root directory.
2.  Run the following command:
    ```bash
    streamlit run app.py
    ```
3.  The app will open in your default browser. You can:
    - Select which models to run from the sidebar (CNN, CNN-LSTM, ViT).
    - Upload an image (JPG, PNG).
    - View the predicted class, confidence score, and probability distribution for each model.

### Training the Models

If you wish to retrain the models or explore the training process:

1.  Open the Jupyter notebooks:
    ```bash
    jupyter notebook
    ```
2.  **CNN & CNN-LSTM**: Open `CNN_and_CNN_LSTM.ipynb`. This notebook covers data loading, preprocessing, and training for both custom architectures.
3.  **Vision Transformer**: Open `vit-freeze.ipynb`. This notebook demonstrates fine-tuning a pre-trained ViT model.

## 🧠 Models Overview

1.  **Simple CNN**: A custom Convolutional Neural Network with 3 convolutional blocks followed by fully connected layers. Good baseline for image classification.
2.  **CNN-LSTM**: A hybrid model that uses CNN layers for feature extraction and an LSTM layer to capture sequential/spatial dependencies in the flattened feature maps.
3.  **Vision Transformer (ViT)**: Uses the `vit_b_16` architecture pre-trained on ImageNet and fine-tuned for this specific dataset. generally offers state-of-the-art performance.

## 📊 Classes

The models are trained to classify images into the following categories:

- Airplane
- Car
- Cat
- Dog
- Flower
- Fruit
- Motorbike
- Person
