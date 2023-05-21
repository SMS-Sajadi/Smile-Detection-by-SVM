# Importing Needed Libraries
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle

import cv2

# Importing my own modules
from Modules.Image_loader import data_loader
from Modules.Face_Detector import face_detector
from Modules.Feature_Extractor import feature_extract
from Modules.Train_Test_Split import data_split


def main():
    # Loading images and their labels
    images, labels = data_loader(GENKI_IMAGE_ROOT, GENKI_LABEL_ROOT)
    print("|----------------------- Dataset Loaded -----------------------|\n")

    # Detecting face in the images
    images = face_detector(images)
    print("|----------------------- Face Detection Done -----------------------|\n")

    # Feature Extracting
    feature_matrix = feature_extract(images)
    print("|----------------------- Features Extracted -----------------------|\n")

    # Splitting the image's features
    train_features, train_labels, test_features, test_labels = data_split(feature_matrix, labels, 0.3, shuffling=True)
    print("|----------------------- Dataset Split Done -----------------------|\n")

    # Training the SVM
    svm_model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
    svm_model.fit(train_features, train_labels)
    print("|----------------------- Model Trained -----------------------|\n")

    # Finding the train accuracy
    print("|----------------------- Training Result -----------------------|")
    train_output = svm_model.predict(train_features)
    train_accuracy = accuracy_score(train_output, train_labels)
    print(f"Accuracy: %{train_accuracy*100} on {len(train_features)} train data\n\n")

    # Testing the Model
    print("|----------------------- Testing Result -----------------------|")
    test_output = svm_model.predict(test_features)
    test_accuracy = accuracy_score(test_output, test_labels)
    print(f"Accuracy: %{test_accuracy*100} on {len(test_features)} train data\n\n")

    # Ask for saving
    user_entry = input("Do you want to save this model?(y,N): ")
    if user_entry == "y":
        # Saving the Model
        with open("Model/SVM_Trained.pkl", "wb") as f:
            pickle.dump(svm_model, f)

        print("-- Model Saved Successfully in ./Model/SVM_Trained.pkl --")

    elif user_entry == "N":
        # Ignoring the trained Model
        print("--- Model didn't saved ---")
    return

N
if __name__ == "__main__":
    # Setting Genki dataset address
    GENKI_IMAGE_ROOT = "../Genki Dataset/genki4k/files/"
    GENKI_LABEL_ROOT = "../Genki Dataset/genki4k/labels.txt"

    main()
