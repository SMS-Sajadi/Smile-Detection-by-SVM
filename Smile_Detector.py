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
        # Reading the frame
        ret, frame = capture.read()

        # Check if frame is read
        if ret:
            # Resizing frame for better showing
            frame = cv2.resize(frame, (480, 720))

            # Detecting all faces in the frame and find their locations
            faces, locs = face_detector_single(frame)

            # Iterating over the faces detected
            for i, face in enumerate(faces):
                # Set the locations of the face detected
                x, y, w, h = locs[i]

                # Extracting the features of the frame
                feature = feature_extract_single(face)

                # Get prediction from the model
                output = svm_model.predict(feature)

                # if smile predicted
                if output[0] == '1':
                    print(1)

                    # Color will be Green
                    color = (0, 255, 0)
                else:
                    print(0)

                    # if smile is not detected, color will be red
                    color = (0, 0, 255)

                # Creating a rectangle with the given locations and color based on smile detection
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Showing the frame with rectangle
            cv2.imshow("Video", frame)
            cv2.waitKey(5)

        else:
            # Frame is not read
            print("-- Frame is not loaded successfully --")
            break


if __name__ == "__main__":

    main()
