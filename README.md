# Anomaly-Detection
Anomaly Detection using opencv
Link to download the dataset:  https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz

Anomaly Detection Using OpenCV and SVM-
This project implements an anomaly detection system using OpenCV for image processing and Support Vector Machines (SVM) for classification. The goal is to identify damaged or anomalous objects in images, such as broken bottles, and differentiate them from normal (good) objects.

Key Features:
> HOG Feature Extraction: The Histogram of Oriented Gradients (HOG) method is used to extract image features for anomaly detection.
> SVM Classification: A pre-trained One-Class SVM model is used to classify images as normal or anomalous.
> Performance Metrics: The model's performance is evaluated using accuracy, precision, and recall.

Python Code- 
1. Anomaly Prediction Function

def predict_anomaly(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    features = hog.compute(img)
    prediction = svm.predict(features.reshape(1, -1))
    return prediction[0] == -1
    
This function takes the image path as input and predicts whether the image is anomalous or normal.
  1. Load Image: The image is loaded in grayscale using OpenCV.
  2. Resize Image: It is resized to 128x128 pixels to ensure uniform feature extraction.
  3. HOG Feature Extraction: Histogram of Oriented Gradients (HOG) features are computed from the image.
  4. SVM Prediction: The extracted HOG features are passed to the pre-trained SVM model, which predicts whether the image is anomalous (-1 for anomaly, 1 for normal).
  5. Return Result: The function returns True if an anomaly is detected (-1), and False otherwise.

2. Evaluating the Model on the Test Dataset
Code- 
test_folder = os.path.join(dataset_path, 'test')
true_labels = []
predicted_labels = []

> The test dataset is located in the test folder, which contains subfolders for each class of images:
  > good: Normal images (no anomalies).
  > broken_large, contamination, etc.: Anomalous images.

> Two lists are initialized to store:
  > true_labels: The actual labels for each image (0 for normal, 1 for anomaly).
  > predicted_labels: The predicted labels from the model (0 for normal, 1 for anomaly).

3. Processing Each Image in the Test Set
Code-
for subfolder in os.listdir(test_folder):
    subfolder_path = os.path.join(test_folder, subfolder)
    for filename in os.listdir(subfolder_path):
        img_path = os.path.join(subfolder_path, filename)
        true_label = 0 if subfolder == 'good' else 1
        is_anomaly = predict_anomaly(img_path)
        predicted_label = 1 if is_anomaly else 0
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

> The code iterates over each image in the test dataset:
  > True Label: The label is assigned based on the folder the image belongs to (0 for normal, 1 for anomaly).
  > Predict Anomaly: The predict_anomaly() function is called to predict if the image is anomalous or normal.
  > Store Labels: Both the true and predicted labels are stored for evaluation.

4. Calculating Performance Metrics
Code- 
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)

  > Accuracy: Measures the overall correctness of the model's predictions (how many total predictions were correct).
  > Precision: Measures how well the model identifies actual anomalies (reduces false positives).
  > Recall: Measures how well the model detects actual anomalies (reduces false negatives).

5. Displaying the Results
Code- 
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
The calculated metrics are printed to give insights into the model's performance.

Project Structure:
  > train/: Contains images of normal objects used for training the SVM model.
  > test/: Contains subfolders with test images for both normal and anomalous objects.
  > ground_truth/: Contains mask images highlighting the damaged areas in the anomalous images.
  > predict_anomaly(): Predicts whether an image is anomalous or not.
  > HOG Feature Extraction: Used to compute features from images.
  > SVM Classifier: A machine learning model used to predict anomalies.

How to Run:
1. Clone the repository:
bash code-
git clone https://github.com/jraju271/anomaly_detection_opencv.git
cd anomaly_detection_opencv

2. Install the required dependencies: Ensure you have OpenCV, NumPy, and Scikit-learn installed:
Code-
pip install opencv-python numpy scikit-learn

3. Train the SVM Model (if not pre-trained):
  > Load the training dataset and extract HOG features for normal images.
  > Train the One-Class SVM model using the extracted features.

4. Run the anomaly detection: Use the provided code to test the model on new images and get predictions.
