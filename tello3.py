from djitellopy import tello
from time import sleep
import cv2
import time

# Connect to the Tello drone
me = tello.Tello()
me.connect()

# Start the video stream
me.streamon()

# Create a directory to store the images (optional)
import os
image_dir = "saved_images"
os.makedirs(image_dir, exist_ok=True)

# Number of images to capture
num_images = 5

# Capture and save images
for i in range(num_images):
    # Get the frame from the Tello drone
    img = me.get_frame_read().frame

    # Display the frame (optional)
    # cv2.imshow("image", img)

    # Save the image with a timestamp in the filename
    timestamp = int(time.time())
    filename = f"{image_dir}/{timestamp}_{i}.jpg"
    cv2.imwrite(filename, img)

    # Wait for one second between capturing images
    sleep(1)

# Stop the video stream and disconnect
me.streamoff()