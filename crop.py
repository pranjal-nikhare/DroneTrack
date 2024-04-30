import os
import face_recognition
from PIL import Image

def crop_faces(image_path, face_locations):
    # Load the image
    image = Image.open(image_path)

    # Iterate over face locations
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Crop the face
        face_image = image.crop((left, top, right, bottom))

        # Save the cropped face as a new file
        output_path = os.path.splitext(image_path)[0] + f"_face_{i+1}.jpg"
        face_image.save(output_path)

if __name__ == "__main__":
    directory_path = "knn_examples/train/vinay_chuu"  # Directory containing images
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            print("Detected face locations in", filename, ":", face_locations)

            crop_faces(image_path, face_locations)
