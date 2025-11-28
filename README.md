🛰️ Aerial Object Classification & Detection

A Deep Learning System for Classifying Birds vs. Drones
(With Optional YOLOv8 Object Detection)

✅ Project Status: Completed Successfully

All tasks including dataset preparation, preprocessing, model training, evaluation, comparison, YOLOv8 detection (optional), and Streamlit deployment have been successfully completed.

The project now includes:
✔ Fully trained Custom CNN model
✔ Fully trained Transfer Learning model (best model selected)
✔ YOLOv8 object detection model (optional)
✔ Streamlit UI for real-time classification/detection
✔ Complete evaluation reports & visualizations
✔ Deployment-ready repository

🎓 Skills Gained

Deep Learning

Computer Vision

Image Classification

Object Detection

TensorFlow / Keras or PyTorch

Data Preprocessing & Augmentation

YOLOv8 (Ultralytics)

Model Evaluation & Visualization

Streamlit Web App Deployment

🌍 Domain Applications

Aerial Surveillance

Wildlife Monitoring

Airport Safety

Military Security & Defense

Environmental Research

📌 Problem Statement

This project builds a deep learning-based system that classifies aerial objects into:

Bird

Drone

The system optionally performs object detection using YOLOv8 to locate and identify objects inside real-world scenes.

This helps in:
✔ Security surveillance
✔ Drone monitoring in restricted airspace
✔ Wildlife protection
✔ Airport bird-strike prevention
✔ Automated monitoring systems

The final solution is deployed using Streamlit, allowing users to upload images and instantly get classification/detection outputs.

📂 Project Workflow (Completed)
1️⃣ Dataset Understanding

✔ Verified folder structure
✔ Checked image distribution per class
✔ Inspected class imbalance
✔ Visualized sample images

2️⃣ Data Preprocessing

✔ Normalized image pixels
✔ Resized all images to 224×224
✔ Converted labels to categorical format

3️⃣ Data Augmentation

Applied transformations to avoid overfitting:

Rotation

Horizontal/Vertical flip

Random zoom

Brightness variation

Random cropping

4️⃣ Model Building
✔ Custom CNN Model

Convolutional + MaxPooling layers

Batch Normalization

Dropout regularization

Dense softmax classifier

✔ Transfer Learning Models

Successfully built & tested:

ResNet50

MobileNet

EfficientNetB0

Fully fine-tuned on the dataset.

5️⃣ Model Training

✔ EarlyStopping implemented
✔ ModelCheckpoint used
✔ Training logs saved

Metrics tracked:

Accuracy

Precision

Recall

F1 Score

6️⃣ Model Evaluation

✔ Confusion matrix
✔ Classification report
✔ Accuracy & loss graphs
✔ Misclassified image analysis

7️⃣ Model Comparison

Compared models on:

Model	Accuracy	F1 Score	Training Time	Generalization

Mobilenet chosen as the best-performing classifier.

Saved as:
best_model.h5 / best_model.pt

🟦YOLOv8 Object Detection (Completed)

✔ Installed YOLOv8
✔ Prepared images + YOLO label TXT files
✔ Created data.yaml
✔ Trained YOLOv8s model
✔ Validated detection performance
✔ Inference tested on sample images
✔ Detection output images saved

🖥️ Streamlit Deployment (Completed)
Features:

✔ Upload an image
✔ See classification: Bird or Drone
✔ View confidence score
✔ (Optional) Run YOLOv8 detection & show bounding boxes

Run the app:
streamlit run app.py

📦 Project Deliverables (All Completed)

✔ Custom CNN trained model

✔ Transfer Learning trained model

✔ YOLOv8 detection model (optional)

✔ Streamlit application

✔ Evaluation graphs (accuracy, loss, confusion matrix)

✔ Inference results on sample images

✔ Fully commented training scripts

✔ Jupyter notebooks for each step

✔ Final report + documentation

🖥️ Model Evaluation

Train accuracy: 0.83
Train_loss: 0.31
Test Accuracy : 0.98
Train_loss: 0.26
