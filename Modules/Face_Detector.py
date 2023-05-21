# Importing Needed Libraries
import cv2


def face_detector(images):
    """
    This function will crop the faces in the images and return the list of them
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


def face_detector_single(img):
    """
    This function will crop the face in the image and return the list of them
    :param img:
    :return faces, loc:
    """

    # Changing image into gray scale
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # A list for saving the faces and their locations
    faces = []
    loc = []

    # Creating the face detector obj
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detecting face in image
    face_detected = face_detector.detectMultiScale(image, 1.1, 4)

    for (x, y, w, h) in face_detected:
        # Location of face
        loc.append((x,y,w,h))

        # Save the detected face
        face_crop = image[y: y + h, x: x + w]
        faces.append(cv2.resize(face_crop, (64,64)))

    return faces, loc
