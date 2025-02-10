import cv2
import mediapipe as mp
import numpy as np



class FaceLandmarks:
    def detect_face_landmarks(image):
        """
        Processes 1 face in the provided image and returns its landmarks
        Landmarks are x, y cordinates normalized between 0 and 1 and then multiplied by img. width and height
        Normalization helps adapt all coordinates to one space, regardless of src. image size
        """
        mp_face_mesh = mp.solutions.face_mesh

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

        landmarks_array = np.array(output)
        flattened_landmarks = landmarks_array.flatten()

        return flattened_landmarks