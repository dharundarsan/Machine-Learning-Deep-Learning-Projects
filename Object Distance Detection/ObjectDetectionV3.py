import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # YOLOv8 pretrained on COCO

# Function to calculate distance from stereo images
def find_dist(pt1, pt2, baseline, f_pixel):
    # Extract pixel coordinates
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]

    # Calculate disparity
    disparity = x1 - x2

    # Avoid division by zero
    if disparity == 0:
        raise ValueError("Disparity is zero; cannot calculate distance.")

    # Calculate distance in the Z direction
    dist_z = (baseline * f_pixel) / disparity

    # Calculate distances in X and Y directions (assuming camera is at origin)
    dist_x = (x1 - 320) * dist_z / f_pixel  # Assuming image width is 640 pixels
    dist_y = (y1 - 240) * dist_z / f_pixel  # Assuming image height is 480 pixels

    return dist_x, dist_y, dist_z

# Function to detect cars and calculate distance
def detect_and_calculate_distance(left_frame, right_frame, baseline, focal_length_mm, alpha):
    # Resize frames to match dimensions
    h_left, w_left, _ = left_frame.shape
    h_right, w_right, _ = right_frame.shape

    if (h_left != h_right) or (w_left != w_right):
        right_frame = cv2.resize(right_frame, (w_left, h_left))

    results_left = model(left_frame)
    results_right = model(right_frame)

    # Check for detections with a confidence threshold
    confidence_threshold = 0.5  # Adjust this value as needed
    car_left = None
    car_right = None

    # Print all detected boxes for debugging purposes
    print("Left Frame Detections:")
    for box in results_left[0].boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Coordinates: {box.xywh}")

    print("Right Frame Detections:")
    for box in results_right[0].boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Coordinates: {box.xywh}")

    # Filter detections based on confidence threshold for left frame
    if len(results_left[0].boxes.xywh) > 0:
        for box in results_left[0].boxes:
            if box.conf > confidence_threshold and box.cls == 0:  # Assuming class '0' is 'car'
                car_left = box.xywh[:2].cpu().numpy()
                break

    # Filter detections based on confidence threshold for right frame
    if len(results_right[0].boxes.xywh) > 0:
        for box in results_right[0].boxes:
            if box.conf > confidence_threshold and box.cls == 0:  # Assuming class '0' is 'car'
                car_right = box.xywh[:2].cpu().numpy()
                break

    # Check if cars were detected successfully
    if car_left is None or car_right is None:
        print("No valid car detected in one or both frames.")
        return None

    # Ensure both car_left and car_right have valid coordinates before proceeding
    if len(car_left) < 2 or len(car_right) < 2:
        print("Detected coordinates do not have enough elements.")
        return None

    # Calculate focal length in pixels
    f_pixel = (w_left * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)

    dist_x, dist_y, dist_z = find_dist(car_left, car_right, baseline, f_pixel)

    return dist_x, dist_y, dist_z, car_left, car_right

# Function to draw bounding boxes on the images and display them
def display_detected_cars(left_frame, right_frame, car_left, car_right):
    # Draw bounding boxes on the left frame
    cv2.rectangle(left_frame, (int(car_left[0] - 50), int(car_left[1] - 25)), 
                    (int(car_left[0] + 50), int(car_left[1] + 25)), 
                    (0, 255, 0), 2)

    # Draw bounding boxes on the right frame
    cv2.rectangle(right_frame, (int(car_right[0] - 50), int(car_right[1] - 25)), 
                    (int(car_right[0] + 50), int(car_right[1] + 25)), 
                    (0, 255, 0), 2)

    # Display the images with detected objects
    cv2.imshow('Left Frame - Detected Car', left_frame)
    cv2.imshow('Right Frame - Detected Car', right_frame)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load stereo images (replace with actual images)
left_frame = cv2.imread('D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\left2.jpg')
right_frame = cv2.imread('D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\right2.jpg')

# Check if images are loaded properly
if left_frame is None or right_frame is None:
    print("Error: Unable to load images.")
else:
    print("Images loaded successfully.")

baseline = 0.064  # Baseline distance between stereo cameras in meters
focal_length_mm = 3.6  # Focal length of the camera in mm
alpha = 70  # Field of view of the camera in degrees

# Calculate the distance of the detected car
distances = detect_and_calculate_distance(left_frame, right_frame, baseline, focal_length_mm, alpha)

if distances:
    dist_x, dist_y, dist_z, car_left_coords, car_right_coords = distances
    print(f"Detected car distance: X = {dist_x:.2f} meters, Y = {dist_y:.2f} meters, Z = {dist_z:.2f} meters")
    
    # Display detected cars on the UI
    display_detected_cars(left_frame.copy(), right_frame.copy(), car_left_coords, car_right_coords)