# Importing Needed Libraries
import cv2
import pickle

# Importing my own moduls
from Modules.Face_Detector import face_detector_single
from Modules.Feature_Extractor import feature_extract_single

def main():
    # Loading the trained model
    with open("./Model/SVM_Trained.pkl", "rb") as f:
        svm_model = pickle.load(f)
        print("-- Model loaded Successfully --")

    # Accessing Camera and writing on it
    capture = cv2.VideoCapture("20230521_142847.mp4")
    #video_writer = cv2.VideoWriter()

    # Reading the Camera
    while(True):
        ret, frame = capture.read()
        frame = cv2.resize(frame, (480, 720))
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector_single(gray_frame)
            for face in faces:
                feature = feature_extract_single(face)

                output = svm_model.predict(feature)

                if output[0] == '1':
                    print(1, end=" ")
                else:
                    print(0, end=" ")
            cv2.imshow("Video", frame)
            cv2.waitKey(10)


if __name__ == "__main__":

    main()
