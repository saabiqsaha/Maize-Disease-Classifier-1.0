# Maize Disease Classification Model 🌽🤖

This project develops a MobileNet-based machine learning model to detect maize diseases, providing actionable insights for farmers.

## Contributors
- Mohammed Saabiq Saha: [@saabiqsaha](https://github.com/Sahastudios1)

## Objectives
- Collect maize disease data.
- Train a MobileNet model for disease identification.
- Build a Streamlit-based frontend.
- Deploy a TensorFlow Lite model in a mobile app.

## Workflow
1. **Data Collection:** Images sourced from Kaggle, split 70% training, 30% testing.
2. **Pre-Processing:** Images resized to 224x224, pixel values normalized, and augmented with flips, rotations, etc.
3. **Model Training:** MobileNet-V2 fine-tuned with custom layers for maize disease classification.

## Model Design
- **Base Model:** MobileNet-V2 for efficiency and accuracy.
- **Custom Layers:** Dense (128 neurons, ReLU), Dropout (50%), Softmax output.
- **Hyperparameters:** Learning rate: 0.001, Batch size: 16, Epochs: 50, Optimizer: Adam.

## Results
- **Training Accuracy:** 93.58%
- **Validation Accuracy:** 91.29%
- **Optimization:** TensorFlow Lite conversion with pruning and quantization reduced model size to 2.4MB.

## Deployment
- **Mobile App:** Built with React Native, integrates TensorFlow Lite for real-time inference.
- **Features:** Image upload, disease prediction with confidence scores, and treatment recommendations.

<div align="center">
  <img src="" alt="App Interface" height="450" width="400">
  <p><em>App User Interface</em></p>
</div>

## Key Visuals
- **Workflow:** ![Machine Learning Workflow](https://drive.google.com/uc?export=view&id=1ZviuEP_vGuBGIhExYLygphSfWbNp1t_W)
- **Sample Dataset Images:** ![Sample Images]()
- **Model Summary:** ![Model Summary]()
- **Training Results:** ![Training Results]()
- **Quantized Model:** ![Quantized Model]()
- **Predictions:** ![Predictions]()

## User Interaction
1. Upload crop leaf images.
2. Real-time inference identifies diseases.
3. Displays predictions with confidence levels.
4. Provides actionable recommendations.

<div align="center">
  <img src="" alt="Inference" height="500" width="400">
</div>



