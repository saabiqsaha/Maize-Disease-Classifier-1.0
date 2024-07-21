Maize Disease Classification Model 🌽🤖
This project involves the development of a machine learning model for detecting diseases in maize plants. The model utilizes a MobileNet architecture for efficient and accurate inference, providing farmers with actionable insights to manage crop health effectively.

Contributors
Mohammed Saabiq Saha : @saabiqsaha
Project Aim and Objectives
Review literature on similar scholarly publications.
Collect data on diseases affecting maize plants.
Train and develop a machine learning model to aid the identification and management of maize diseases.
Build a user-friendly frontend using Streamlit.
Test and validate the model's performance.
Methodology and Implementation
Every general machine learning task, including deep learning, follows a standardized approach to obtain the best results. This is known as the Machine Learning workflow, which includes data collection, pre-processing, model training, and evaluation.

Data Collection
Description of Dataset

The dataset used for this project consists of images of maize plants affected by various diseases. The data was split into training (80%) and validation/testing (20%) subsets to ensure robust model performance.

Data Pre-Processing and Augmentation
Color images from the dataset were resized to 224x224 pixels to meet the input requirements of the MobileNet model. Pixel values were rescaled from [0, 255] to [0, 1] to aid neural network training. To prevent overfitting, data augmentation techniques such as random flips, mirror images, and rotations up to 20% were applied. These augmentations were generated on-the-fly during training to enhance model robustness and accuracy in classifying real-world images.

Model Selection and Design
Choice of MobileNet
MobileNet was chosen for its efficiency and suitability for deployment on devices with limited computational power. Its design uses depth-wise separable convolutions, reducing computation while maintaining high accuracy.

Customizing the Top Layers
The top layers were fine-tuned to recognize maize diseases by replacing the original classification head with densely connected layers, tailored for our specific task.

Hyperparameter Tuning
Key hyperparameters were fine-tuned to optimize performance:

Learning Rate: Adjusted to balance between fast convergence and avoiding divergence.
Batch Size: Selected based on computational resources and dataset characteristics.
Optimization Algorithm: Used Adam for adaptive learning.
Early Stopping: Implemented to prevent overfitting.
Learning Rate Schedules: Applied to help the model converge to a better solution.
Model Architecture
Base Model: MobileNet
Custom Layers:
Flatten Layer: Converts the 2D feature maps into a 1D feature vector.
Dense Layer: Includes 128 neurons with ReLU activation.
Dropout Layer: Added for regularization with a dropout rate of 50%.
Output Layer: Utilizes softmax activation to classify the different maize diseases.
Training and Evaluation
Training and evaluation were conducted on Google Colab, leveraging its cloud-based environment for enhanced computational power. The model was trained over 50 epochs with early stopping to prevent overfitting. By the final epoch, the model achieved satisfactory training and validation accuracies.

Frameworks Used
Streamlit: Used for the frontend to run the model in the browser. Streamlit is popular among data scientists for analyzing and visualizing data.
Potential Benefits
This model can be deployed on mobile devices, making it accessible to farmers in rural areas. It helps them quickly and accurately identify diseases in their crops, potentially saving crops and ensuring better yields.

Future Work
While the current model shows promising results, it has shown signs of overfitting. Future work will focus on creating more balanced models and improving generalization performance.

Contributing
Contributions and collaborations to improve this project are welcome. Please feel free to fork the repository and submit pull requests.

Special Thanks
Special thanks to Sascha Dittmann and Felipe Moreno for their tutorials that helped immensely in building this project.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For any questions or further information, please contact Mohammed Saabiq Saha.
