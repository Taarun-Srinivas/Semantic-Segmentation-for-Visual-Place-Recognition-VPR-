import cv2
import os

# Path to the directory containing image frames
# image_dir = r"C:\resources\AI4CE lab works\RESULTS\wet_cloudy_data\cam1\cluster_pred"
image_dir = r"C:\resources\AI4CE lab works\carla_vrp_data_gen\rain_noon_data\cam2"

# Get a list of all the image files in the directory
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")])

# Load the first image to get the dimensions
image = cv2.imread(os.path.join(image_dir, image_files[0]))
height, width, _ = image.shape

# Define the output video filename and codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose the codec you want to use, e.g., 'XVID', 'MJPG', 'mp4v', etc.
output_filename = r"C:\resources\AI4CE lab works\rain_noon_input.mp4"

# Create the video writer object
video_writer = cv2.VideoWriter(output_filename, fourcc, 25.0, (width, height))  # Adjust frame rate as needed

# Loop through all the image files and write them to the video
for image_file in image_files:
    # Load the image
    image = cv2.imread(os.path.join(image_dir, image_file))

    # Write the image to the video
    video_writer.write(image)

# Release the video writer
video_writer.release()

print(f"Video saved as {output_filename}")