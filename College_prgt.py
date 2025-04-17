

# Import necessary libraries
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import classification_report
# Define a function to extract features from audio files
def extract_features(file_path):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Define the path to the dataset
data_path = "/content/drive/MyDrive/Data_set"

# Initialize empty lists for features and labels
features = []
labels = []

# Loop through each subdirectory and extract features from audio files
for subdir, _, files in os.walk(data_path):
    for file in files:
        file_path = os.path.join(subdir, file)
        label = subdir.split("/")[-1]
        features.append(extract_features(file_path))
        labels.append(label)

# Convert the lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Initialize a support vector machine classifier
svm = SVC(kernel='linear')

model =  svm.fit(X_train, y_train)

joblib.dump(model, 'svm_lightning.pkl')

# Predict labels for the testing set
y_pred = model.predict(X_test)
# Evaluate the performance of the classifier using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test,y_pred))

# Load model from save file
svm_model = joblib.load('svm_lightning.pkl')

# Predicting if Thunder is present in a given audio clip
data_loc = "/content/beautiful-random-minor-arp-119378 (1).mp3"
extract_features(data_loc)
New_data_features = np.array(extract_features(data_loc))
print(New_data_features)
New_data_features1=New_data_features.reshape(-20,20)
y_pred = svm_model.predict(New_data_features1)
print(y_pred)
for i in y_pred:
  if i=="Thunder":
    print("Thunderstorm Detected")
  else:
    print("Thunderstorm not Detected")

