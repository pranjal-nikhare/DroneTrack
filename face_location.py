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
        face_image.save(f"face_{i+1}.jpg")

if __name__ == "__main__":
    image_path = "knn_examples/test/day11.jpg"
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    print("Detected face locations:", face_locations)

    crop_faces(image_path, face_locations)