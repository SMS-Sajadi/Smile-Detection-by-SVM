# Importing Needed Libraries
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import cv2

# Importing my own modules
from Modules.Image_loader import data_loader
from Modules.Face_Detector import face_detector
from Modules.Feature_Extractor import feature_extract
from Modules.Train_Test_Split import data_split


def main():
    # Loading images and their labels
    images, labels = data_loader(GENKI_IMAGE_ROOT, GENKI_LABEL_ROOT)

    # Detecting face in the images
    images = face_detector(images)

    # Feature Extracting
    feature_matrix = feature_extract(images)

    # Splitting the image's features
    train_features, train_labels, test_features, test_labels = data_split(feature_matrix, labels, 0.3, shuffling=True)

    # Train the SVM
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
    clf.fit(train_features, train_labels)

    # Sample Test
    output = clf.predict(test_features)
    score = accuracy_score(output, test_labels)
    print(score)

if __name__ == "__main__":
    # Setting Genki dataset address
    GENKI_IMAGE_ROOT = "../Genki Dataset/genki4k/files/"
    GENKI_LABEL_ROOT = "../Genki Dataset/genki4k/labels.txt"

    main()
