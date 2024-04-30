import os
import pickle
from face_recognition.face_recognition_cli import image_files_in_folder
import face_recognition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import imgaug.augmenters as iaa
import cv2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def face_alignment(image):
    face_landmarks = face_recognition.face_landmarks(image)
    if len(face_landmarks) == 0:
        return image  # Return the original image if no face landmarks are detected

    left_eye = np.array(face_landmarks[0]['left_eye'])
    right_eye = np.array(face_landmarks[0]['right_eye'])

    # Ensure left and right eye arrays are 2D
    left_eye = left_eye.reshape(-1, 2)
    right_eye = right_eye.reshape(-1, 2)

    # Calculate the angle between the eyes
    dY = right_eye[0][1] - left_eye[0][1]
    dX = right_eye[0][0] - left_eye[0][0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # Rotate the image to align the eyes horizontally
    center = tuple(np.array(image.shape[1::-1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return aligned_image


def data_augmentation(image):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Affine(rotate=(-10, 10)),  # random rotations
        iaa.GaussianBlur(sigma=(0, 3.0)),  # random Gaussian blur
    ])
    augmented_image = seq(image=image)
    return augmented_image

def normalize_pixel_values(image):
    normalized_image = image / 255.0
    return normalized_image

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image.astype('uint8')

    if image is None:
        raise ValueError("Failed to load image at path: {}".format(image_path))

    # Convert the image to RGB format if it's in BGR format
    if image.shape[-1] != 3:  # Check if the image has 3 channels (RGB or BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face alignment
    aligned_image = face_alignment(image)
    
    # Perform data augmentation
    augmented_image = data_augmentation(aligned_image)
    
    # Normalize pixel values
    normalized_image = normalize_pixel_values(augmented_image)
    
    return normalized_image


def train(train_dir, model_save_path=None, n_neighbors=3, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = preprocess_image(img_path)
            face_encoding = face_recognition.face_encodings(image)
            
            if len(face_encoding) == 0:
                print("No faces found!")
            else:
                face_encoding = face_encoding[0]

            if len(face_encoding) != 1:
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_encoding) < 1 else "Found more than one face"))
            else:
                X.append(face_encoding[0])
                y.append(class_dir)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X_train, y_train)
    
    accuracy = knn_clf.score(X_test, y_test)
    print("Accuracy:", accuracy)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

if __name__ == "__main__":
    print("Training KNN classifier...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=3)
    print("Training complete!")
