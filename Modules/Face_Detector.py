# Importing Needed Libraries
import cv2


def face_detector(images):
    """
    This function will crop the face in the image and return the list of them
    :param images:
    :return faces:
    """

    # A list for saving the faces
    faces = []

    # Creating the face detector obj
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Iterating over the images for face detection
    for img in images:
        # Detecting face in image
        face_detected = face_detector.detectMultiScale(img, 1.1, 4)

        for (x, y, w, h) in face_detected:
            pass

        # Save the detected face
        face_crop = img[y: y + h, x: x + w]
        faces.append(cv2.resize(face_crop, (64,64)))

    return faces
