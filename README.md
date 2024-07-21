# Maize Disease Classification Model 🌽🤖
This project involves the development of a machine learning model for detecting diseases in maize plants. The model utilizes a MobileNet architecture for efficient and accurate inference, providing farmers with actionable insights to manage crop health effectively.

## Contributors
- Mohammed Saabiq Saha : [@saabiqsaha](https://github.com/Sahastudios1)


## Project Aim and Objectives
- Collect data on diseases affecting maize plants.
- Train and develop a machine learning model to aid the identification and management of maize diseases.
- Build a user-friendly frontend using Streamlit.
- Test and validate the model's performance.


## IMPLEMENTATION
Machine learning projects, including those involving deep learning, adhere to a structured process to achieve optimal results. This process, known as the Machine Learning workflow, encompasses steps such as data collection, data pre-processing, model training, and evaluation. 
![Machine Learning Workflow image](https://drive.google.com/uc?export=view&id=1ZviuEP_vGuBGIhExYLygphSfWbNp1t_W)



### DATA COLLECTION
**Description of Dataset**
The dataset for this project, sourced from Kaggle, includes images of maize plants with various diseases. The data was split using a train-test split method, allocating 70% for training and 30% for validation/testing to ensure robust model performance.
![Sample images from the Dataset](https://drive.google.com/uc?export=view&id=10XGh64TyRWrzTbSn01QD-EPCK0SpiFiZ)
*Sample images from the PlantVillage Dataset*



## Data Pre-Processing and Augmentation
Color images from the dataset were resized to 224x224 pixels to meet the input requirements of the MobileNet model. Pixel values were rescaled from [0, 255] to [0, 1] to aid neural network training. To prevent overfitting, data augmentation techniques such as random flips, mirror images, and rotations up to 20% were applied. These augmentations were generated on-the-fly during training to enhance model robustness and accuracy in classifying real-world images.


## Model Selection and Design

### Choice of MobileNet-V2
MobileNet-V2 was chosen for its efficiency and suitability for mobile devices. Its design uses depth-wise separable convolutions, reducing computation while maintaining high accuracy.

### Customizing the Top Layers
The top layers were fine-tuned to recognize maize diseases by replacing the original classification head with densely connected layers, tailored for our specific task.


### Hyperparameter Tuning
Key hyperparameters were fine-tuned to optimize performance:
- **Learning Rate:** 0.001
- **Batch Size:** 16
- **Epochs:** 50
- **Optimization Algorithm:** Adam
- **Early Stopping:** Implemented to prevent overfitting.

### Model Architecture
Base Model: MobileNet
**Custom Layers:**
**Flatten Layer:** Converts the 2D feature maps into a 1D feature vector.
**Dense Layer:** Includes 128 neurons with ReLU activation.
**Dropout Layer:** Added for regularization with a dropout rate of 50%.
**Output Layer:** Utilizes softmax activation to classify the different maize diseases.

The details are listed in the table below:
![Hyperparameter specifications](https://drive.google.com/uc?export=view&id=1zyeDadSd_GdbF-oUyM_C3W-vXNIyIOL-)


A summary of the entire Network Architecture is shown below:
![Model summary](https://drive.google.com/file/d/1t0aQV7IBtZxBgGIx-yc6o0fJV0lnXjrP/view?usp=drive_link)



## Training and Evaluation

Training and evaluation were conducted on Google Colab, leveraging its cloud-based environment for enhanced computational power. The model was trained over 20 epochs with early stopping to prevent overfitting. By the 20th epoch, the model achieved training and validation accuracies of 93.58% and 91.29%, respectively.

![model evaluation](https://drive.google.com/uc?export=view&id=1TKNHL1yXLjEEsJwmRuWnkcnJeQC8cNuV)

*Model Training and Validation Accuracies (left) and Training and Validation loss (right) as a function of epoch number*



Additionally, while validation accuracy offers insight into how the model generalizes to unfamiliar data, the ultimate assessment lies in predictions made on unseen data (test data). Graphical depictions of the model’s predictions on a subset of the test data are provided below, accompanied by the model’s confidence score for each prediction.
![Model predictions](https://drive.google.com/uc?export=view&id=130pyPQj6NIJrfLyvihq6Etb4C6TrHoSP)



### Conversion to TensorFlow Lite
After training, the model was converted to TensorFlow Lite for optimal deployment on mobile and edge devices. This conversion preserved accuracy while enhancing performance and memory efficiency.

- **Pruning:** Removed less important connections, reducing model size and accelerating inference without compromising accuracy.
- **Quantization:** Reduced the precision of numerical values, shrinking the model from 8MB to approximately 2.4MB, speeding up inference and reducing memory demands.

![Quantized Model](https://drive.google.com/uc?export=view&id=1QI1i7SySZnmgJ8gdiuaUpPe7fRSInZV5)

### Model Verification
Post-conversion, the model’s accuracy was validated to ensure the optimization procedures did not adversely affect its proficiency in identifying crop diseases. The TensorFlow Lite conversion process resulted in a lightweight yet accurate model suitable for mobile applications.


## Mobile App Deployment
The deployment of the crop disease identification solution culminated in a user-friendly mobile app built using React Native into which the TensorFlow Lite model was integrated. 

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1g5qHpPwDmXyO1uaXyYGREtKADuzw1M_a" alt="Interface" height = "450" width="400">
  <p><em>App User Interface</em></p>
</div>

### User Interaction and Inference
- Image Upload: Users can upload images of crop leaves directly from their bobile devices
- Real-Time Inference: The TensorFlow Lite model analyzes the image to identify potential diseases.

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1OUkugtCPtTCXfs5UuvzZImNHsU9oo7Qy" alt="Inferfence" height = "500" width="400">
</div>


- Prediction Display: Shows detected diseases with confidence levels in clear format

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1AI7aGpPiYLaiEATqFV34K7Rb5ngvzFjm" alt="predict" height = "500" width="400">
</div>


- Guidance and Recommendations: Offers practical steps for disease management and treatment

 <div align="center">
   <img src="https://drive.google.com/uc?export=view&id=1yOq99VvbRlVW6Zt43fn4kxDlqEzJGptm" alt="Interface" height = "550" width="400">
 </div>



