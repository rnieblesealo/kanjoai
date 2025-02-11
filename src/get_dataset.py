import kagglehub
import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh


def detect_face_landmarks(image):
    """
    Processes 1 face in the provided image and returns its landmarks
    Landmarks are x, y cordinates normalized between 0 and 1 and then multiplied by img. width and height
    Normalization helps adapt all coordinates to one space, regardless of src. image size
    """

    output = []

    # Convert BGR to RGB (required by MediaPipe)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize Face Mesh model
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            print("No face detected!")
            return []

        h, w, _ = image.shape  # Get new image dimensions

        # Process landmarks
        for face_landmarks in results.multi_face_landmarks:
            for i, landmark in enumerate(face_landmarks.landmark):
                # Convert to pixel coordinates
                x, y = int(landmark.x * w), int(landmark.y * h)
                output.append((x, y))
    return output


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

                    landmarks = detect_face_landmarks(image)  # Detect landmarks

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
folders = ["angry", "neutral"]
output_path = "../data/face_landmarks.csv"

process_dataset(
    dataset_path,
    folders,
    output_path
)

