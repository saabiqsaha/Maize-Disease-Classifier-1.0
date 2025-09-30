# Maize Disease Classification Model 🌽🤖

I built a deep learning application to help farmers detect diseases in maize plants using computer vision. The model was trained on thousands of images of healthy and diseased maize leaves to provide instant, accurate diagnoses.

## Contributors
- Mohammed Saabiq Saha: [@saabiqsaha](https://github.com/saabiqsaha)

## Overview
This project uses MobileNet-V2 to classify maize plant diseases, enabling farmers to quickly identify crop health issues and take appropriate action.

## Objectives
- Collect and preprocess maize disease image data
- Train a lightweight MobileNet model for disease identification
- Build an intuitive Streamlit web interface
- Deploy a TensorFlow Lite model for mobile use

## Workflow
1. **Data Collection:** Images sourced from Kaggle, split 70% training, 30% validation
2. **Pre-Processing:** Images resized to 224x224, normalized, and augmented (flips, rotations, zoom)
3. **Model Training:** Fine-tuned MobileNet-V2 with custom classification layers

## Model Architecture
- **Base Model:** MobileNet-V2 (pre-trained on ImageNet)
- **Custom Layers:** 
  - Dense layer (128 neurons, ReLU activation)
  - Dropout (50% to prevent overfitting)
  - Softmax output layer for multi-class classification
- **Hyperparameters:** 
  - Learning rate: 0.001
  - Batch size: 16
  - Epochs: 50
  - Optimizer: Adam

## Results
- **Training Accuracy:** 93.58%
- **Validation Accuracy:** 91.29%
- **Model Size:** 2.4MB (after TensorFlow Lite optimization with pruning and quantization)

## Deployment
- **Web App:** Built with Streamlit for easy farmer access
- **Mobile App:** React Native application with TensorFlow Lite integration
- **Features:** 
  - Image upload functionality
  - Real-time disease prediction with confidence scores
  - Treatment recommendations based on diagnosis

## How It Works
1. Farmer uploads image of maize leaf
2. Model performs real-time inference
3. System displays disease prediction with confidence level
4. Provides actionable treatment recommendations

## Key Visuals
- **Workflow:** ![Machine Learning Workflow](https://drive.google.com/uc?export=view&id=1ZviuEP_vGuBGIhExYLygphSfWbNp1t_W)
- **Sample Dataset Images:** ![Sample Images]()
- **Model Summary:** ![Model Summary]()
- **Training Results:** ![Training Results]()
- **Quantized Model:** ![Quantized Model]()
- **Predictions:** ![Predictions]()

---

**Live Demo:** [Try the Streamlit App](https://maize-disease-classifier.streamlit.app/)