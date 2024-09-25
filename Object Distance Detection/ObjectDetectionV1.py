import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # Load YOLOv8 pretrained on COCO dataset

# Function to calculate distance from stereo images
def find_dist(pt1, pt2, baseline, f_pixel):
    x1, x2 = pt1[0], pt2[0]
    disparity = x1 - x2

    # Check for zero disparity
    if disparity == 0:
        print("Warning: Disparity is zero; cannot calculate distance.")
        return None, None, None

    dist_z = (baseline * f_pixel) / disparity
    dist_x = (x1 - (640 // 2)) * dist_z / f_pixel  # X-coordinate distance
    dist_y = (pt1[1] - (480 // 2)) * dist_z / f_pixel  # Y-coordinate distance
    
    return dist_x, dist_y, dist_z

# Function to calculate IOU between two bounding boxes
def calculate_iou(bbox_left, bbox_right):
    # Extract coordinates for left and right bounding boxes
    x1_left, y1_left, w1_left, h1_left = bbox_left
    x1_right, y1_right, w1_right, h1_right = bbox_right
    
    # Calculate the (x, y) coordinates of the intersection rectangle
    x_left = max(x1_left, x1_right)
    y_top = max(y1_left, y1_right)
    x_right = min(x1_left + w1_left, x1_right + w1_right)
    y_bottom = min(y1_left + h1_left, y1_right + h1_right)
    
    # Compute the area of the intersection rectangle
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    
    # Compute the area of both bounding boxes
    area_left = w1_left * h1_left
    area_right = w1_right * h1_right
    
    # Compute the Intersection Over Union (IOU)
    iou = intersection_area / float(area_left + area_right - intersection_area)
    return iou

# Function to detect cars and calculate distance
def detect_and_calculate_distance(left_frame, right_frame, baseline, focal_length_mm, alpha, iou_threshold=0.5):
    results_left = model(left_frame)  # Detect objects in left image
    results_right = model(right_frame)  # Detect objects in right image
    
    # Extract detected car labels and coordinates from both frames
    cars_left = results_left[0].boxes.xywh.cpu().numpy()
    cars_right = results_right[0].boxes.xywh.cpu().numpy()
    
    # Get car class labels
    labels_left = results_left[0].boxes.cls.cpu().numpy()
    labels_right = results_right[0].boxes.cls.cpu().numpy()
    
    # Set to store detected car names (e.g., 0 for 'car')
    detected_cars_left = {int(label) for label in labels_left if label == 0}
    detected_cars_right = {int(label) for label in labels_right if label == 0}

    # Find intersection of detected cars in both frames
    common_cars = detected_cars_left.intersection(detected_cars_right)
    
    distances = []
    
    for i, car_left in enumerate(cars_left):
        if labels_left[i] in common_cars:
            # Find corresponding car in the right image (based on IOU matching)
            for j, car_right in enumerate(cars_right):
                if labels_right[j] == labels_left[i]:  # Match based on label
                    # Check if IOU of the car bounding boxes is greater than the threshold
                    iou = calculate_iou(car_left, car_right)
                    if iou > iou_threshold:
                        f_pixel = (640 * 0.5) / (np.tan(alpha * 0.5 * np.pi / 180))  # Calculate focal length in pixels
                        dist_x, dist_y, dist_z = find_dist(car_left, car_right, baseline, f_pixel)
                        distances.append((labels_left[i], dist_x, dist_y, dist_z))
                        break  # Exit inner loop after finding match
    
    return distances

# Load stereo images (replace with actual images)
left_frame = cv2.imread('D:\\mini project\\left3.png')
right_frame = cv2.imread('D:\\mini project\\right3.png')

# Check if images are loaded properly
if left_frame is None or right_frame is None:
    print("Error: Unable to load images.")
else:
    print("Images loaded successfully.")

# Parameters for distance calculation
baseline = 0.06  # Baseline distance between stereo cameras in meters
focal_length_mm = 24  # Focal length of the camera in mm
alpha = 70  # Field of view of the camera in degrees

# Calculate the distance of detected cars
distances = detect_and_calculate_distance(left_frame, right_frame, baseline, focal_length_mm, alpha)

# Print distances of detected cars
if distances:
    for label, dist_x, dist_y, dist_z in distances:
        print(f"Detected car label: {label}, Distances -> X: {dist_x:.2f} m, Y: {dist_y:.2f} m, Z: {dist_z:.2f} m")
else:
    print("No common cars detected in both images.")
