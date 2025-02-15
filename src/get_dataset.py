import kagglehub
import cv2
import numpy as np
import os
import csv
from landmarks import FaceLandmarks

def process_dataset(dataset_path, subfolder_names, csv_output_path):
    """
    Process batch of images,
    detecting facial landmarks to each and saving results to CSV file.
    """

    # Open the CSV file in write mode
    with open(csv_output_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        for folder in ['test', 'train']:
            test_and_train_path = os.path.join(dataset_path, folder)

            # Loop through each subfolder (emotion or category)
            for subfolder_name in subfolder_names:
                if subfolder_name == "angry":
                    label = 1
                else:
                    label = 0

                subfolder_path = os.path.join(test_and_train_path, subfolder_name)

                # Check if subfolder exists
                if not os.path.exists(subfolder_path):
                    print(f"Subfolder {subfolder_name} not found.")
                    continue

                # Loop through all images in the subfolder
                for filename in os.listdir(subfolder_path):
                    image_path = os.path.join(subfolder_path, filename)
                    image = cv2.imread(image_path)

                    # Check if the image is valid
                    if image is None:
                        continue

                    # preview image
                    # cv2.imshow("image", image)
                    # cv2.waitKey(0)

                    face_landmarks_detector = FaceLandmarks()  # Create an instance
                    landmarks = face_landmarks_detector.detect_face_landmarks(image=image)  # Call the method properly

                    # If no landmarks are detected, skip the image
                    if len(landmarks) == 0:
                        continue

                    landmarks_array = np.array(landmarks)
                    flattened_landmarks = landmarks_array.flatten()

                    # Write the dynamic label followed by each landmark as a separate column
                    row = [label]
                    # Append all the landmark values
                    row.extend(flattened_landmarks)
                    writer.writerow(row)

    print(f"CSV file has been saved to {csv_output_path}")





# download straight from kaggle!
dataset_path = f"{kagglehub.dataset_download('msambare/fer2013')}"
folders = []

folders.append("angry")
folders.append("disgust")
# folders.append("fear")
folders.append("happy")
# folders.append("neutral")
# folders.append("sad")
folders.append("surprise")

output_path = "../data/face_landmarks.csv"

process_dataset(
    dataset_path,
    folders,
    output_path
)

# Need more angry images

