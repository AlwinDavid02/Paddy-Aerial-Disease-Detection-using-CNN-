mport cv2
import requests
import os
import time

# URL of the video feed
video_feed_url = "http://192.168.19.8:5000/video_feed"  # Adjust the URL as needed

# Directory to save downloaded images
output_directory = r"C:\Users\alwin\Desktop\Alwin\matlab"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

def download_frames(url, output_dir):
    # Open the video feed
    cap = cv2.VideoCapture(url)

    # Initialize frame count
    frame_count = 0

    # Loop through the frames
    while True:
        # Read a frame from the video feed
        ret, frame = cap.read()

        # Break the loop if no frame is retrieved
        if not ret:
            break

        # Save the frame as an image
        image_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(image_path, frame)

        # Increment frame count
        frame_count += 1

        # Wait for 0.5 seconds (500 milliseconds)
        #cv2.waitKey(500)
        time.sleep(0.5)

    # Release the video capture object
    cap.release()

    print(f"Downloaded {frame_count} frames.")

if __name__ == "__main__":
    # Download frames from the video feed
    download_frames(video_feed_url, output_directory)