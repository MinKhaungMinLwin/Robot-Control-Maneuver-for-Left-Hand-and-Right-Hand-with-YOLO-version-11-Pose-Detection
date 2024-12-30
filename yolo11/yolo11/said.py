import cv2
import os
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 Pose model
model = YOLO('yolo11x_pose.pt')

# Create directories for saving poses
os.makedirs('poses/left', exist_ok=True)
os.makedirs('poses/right', exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(1)  # Change 0 to your webcam index if needed
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

tolerance = 10  # Distance threshold for determining horizontal alignment
frame_count = 0  # To name saved frames uniquely

# Pose detection and saving logic
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Run pose estimation
    results = model(frame)

    # Process each detected person
    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.cpu().numpy()

            if keypoints.shape[0] > 0:  # Ensure keypoints are detected
                for person_kp in keypoints:
                    if person_kp.shape[0] >= 17:  # Ensure at least 17 keypoints are available
                        left_wrist, right_wrist = person_kp[9], person_kp[10]
                        left_elbow, right_elbow = person_kp[7], person_kp[8]
                        print(left_wrist)
                        # Confidence threshold for left side
                        if left_wrist[2] > 0.5 and left_elbow[2] > 0.5:
                            # Check if left wrist and left elbow are horizontally aligned
                            if abs(left_wrist[1] - left_elbow[1]) <= tolerance:
                                save_path = f"poses/left/horizontal_left_{frame_count}.jpg"
                                if cv2.imwrite(save_path, frame):
                                    print(f"Saved horizontally aligned left pose: {save_path}")
                                else:
                                    print(f"Error saving horizontal left pose: {save_path}")
                                frame_count += 1

                        # Confidence threshold for right side
                        if right_wrist[2] > 0.5 and right_elbow[2] > 0.5:
                            # Check if right wrist and right elbow are horizontally aligned
                            if abs(right_wrist[1] - right_elbow[1]) <= tolerance:
                                save_path = f"poses/right/horizontal_right_{frame_count}.jpg"
                                if cv2.imwrite(save_path, frame):
                                    print(f"Saved horizontally aligned right pose: {save_path}")
                                else:
                                    print(f"Error saving horizontal right pose: {save_path}")
                                frame_count += 1

    # Visualize keypoints on frame
    annotated_frame = results[0].plot() if hasattr(results[0], 'plot') else frame
    cv2.imshow('Pose Detection', annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()