# Importing Needed Libraries
import numpy as np
from skimage.feature import hog, local_binary_pattern


def feature_extract(images):
    """
    This function will extract hog and lbp feature of images and return the feature matrix
    :param images:
    :return image_features matrix:
    """
    # A list for saving features
    image_features = []
    radius = 3
    n_points = 8 * radius
    for image in images:
        # Extracting HOG of image
        hog_features = hog(image, orientations=16, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False,
                           feature_vector=True)

        # Extracting LBP of image
        lbp_features = local_binary_pattern(image, n_points, radius, method='default')

        # Adding features to the matrix
        image_features.append(np.concatenate((hog_features, lbp_features.flatten())))

    # Change list to numpy array
    image_features = np.array(image_features)

    return image_features
