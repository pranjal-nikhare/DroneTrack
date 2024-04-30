import requests
from testing import predict
import os

# Function to get the image from the API link
def get_image_from_api(api_link, output_filename):
    response = requests.get(api_link)
    if response.status_code == 200:
        with open(output_filename, 'wb') as f:
            f.write(response.content)
        return output_filename
    else:
        print("Failed to fetch image from API:", response.status_code)
        return None

# Function to process the image using the testing script and save the output to a text file
def process_image_and_save_output(image_path, output_filename):
    predictions = predict(image_path, model_path="trained_knn_model.clf")
    with open(output_filename, 'w') as f:
        for name, (top, right, bottom, left) in predictions:
            f.write("- Found {} at ({}, {})\n".format(name, left, top))

if __name__ == "__main__":
    # Define the API link
    api_link = "http://example.com/image.jpg"

    # Define the output filenames
    image_filename = "test_image.jpg"
    output_filename = "output.txt"

    # Get the image from the API link
    image_path = get_image_from_api(api_link, image_filename)

    if image_path:
        print("Image fetched successfully from API:", image_path)

        # Process the image and save the output
        process_image_and_save_output(image_path, output_filename)

        print("Output saved to:", output_filename)

        # Optionally, remove the image file after processing
        os.remove(image_path)
