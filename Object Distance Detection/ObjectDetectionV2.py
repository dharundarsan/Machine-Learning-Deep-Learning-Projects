import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # YOLOv8 pretrained on COCO

# Function to calculate Intersection over Union (IoU) for bounding boxes
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the (x, y)-coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # If there is no overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Compute the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Function to check if the detected car in the left image is the same as in the right image using IoU and color
def is_same_car(box_left, box_right, left_frame, right_frame, iou_threshold=0.5, color_threshold=50):
    if box_left is None or box_right is None:
        return False

    # IoU comparison
    iou = calculate_iou(box_left, box_right)
    
    # Crop the car regions from both images
    car_left_region = left_frame[int(box_left[1]):int(box_left[1] + box_left[3]), int(box_left[0]):int(box_left[0] + box_left[2])]
    car_right_region = right_frame[int(box_right[1]):int(box_right[1] + box_right[3]), int(box_right[0]):int(box_right[0] + box_right[2])]

    # Calculate the average color of the car in both images
    avg_color_left = np.mean(car_left_region, axis=(0, 1))  # Average color of left car
    avg_color_right = np.mean(car_right_region, axis=(0, 1))  # Average color of right car

    # Color difference
    color_diff = np.linalg.norm(avg_color_left - avg_color_right)

    # Check if IoU and color difference are within thresholds
    return iou >= iou_threshold and color_diff < color_threshold

# Function to calculate distance from stereo images
def find_dist(pt1, pt2, baseline, f_pixel, alpha):
    disparity = pt1[0] - pt2[0]
    if disparity <= 0:
        return None, None, None  # No valid disparity

    # Calculate the Z distance (depth)
    dist_z = (baseline * f_pixel) / disparity

    # Calculate the X and Y distances
    dist_x = (pt1[0] - (f_pixel / 2)) * dist_z / f_pixel
    dist_y = (pt1[1] - (f_pixel / 2)) * dist_z / f_pixel

    return dist_x, dist_y, dist_z

# Function to detect cars and calculate distance
def detect_and_calculate_distance(left_frame, right_frame, baseline, focal_length_mm, alpha):
    # Resize frames to match dimensions, if needed
    if left_frame.shape != right_frame.shape:
        right_frame = cv2.resize(right_frame, (left_frame.shape[1], left_frame.shape[0]))

    results_left = model(left_frame)
    results_right = model(right_frame)

    # Check if there are any detections in both images
    if len(results_left[0].boxes) == 0 or len(results_right[0].boxes) == 0:
        print("No cars detected in one of the images.")
        return None, None, None, False

    # Find the first car detected (YOLO class 2 is car)
    car_left = None
    car_right = None

    for box, cls in zip(results_left[0].boxes.xywh, results_left[0].boxes.cls):  # Iterating over left image boxes
        if int(cls) == 2:  # Car class (index 2)
            car_left = box.cpu().numpy()
            break

    for box, cls in zip(results_right[0].boxes.xywh, results_right[0].boxes.cls):  # Iterating over right image boxes
        if int(cls) == 2:  # Car class (index 2)
            car_right = box.cpu().numpy()
            break

    if car_left is None or car_right is None:
        print("No car detected in one of the images.")
        return None, None, None, False

    # Check if the detected cars in left and right images are the same using IoU and color
    if is_same_car(car_left, car_right, left_frame, right_frame):
        print("Same car detected in both images.")
        # Use the center of the bounding box for distance calculation
        dist_x, dist_y, dist_z = find_dist(car_left[:2], car_right[:2], baseline, focal_length_mm, alpha)
        return dist_x, dist_y, dist_z, True
    else:
        print("Different cars detected in the images.")
        return None, None, None, False

# Load stereo images (replace with actual images)
left_frame = cv2.imread('D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\left1.jpg')
right_frame = cv2.imread('D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\right1.jpg')

# Check if images are loaded properly
if left_frame is None or right_frame is None:
    print("Error: Unable to load images.")
else:
    print("Images loaded successfully.")

baseline = 0.064  # Baseline distance between stereo cameras in meters
focal_length_mm = 3.6  # Focal length of the camera in mm
alpha = 70  # Field of view of the camera in degrees

# Calculate the distance of the detected car and check if they are the same
dist_x, dist_y, dist_z, same_car = detect_and_calculate_distance(left_frame, right_frame, baseline, focal_length_mm, alpha)

if same_car:
    print(f"Detected car distance: X = {dist_x:.2f} meters, Y = {dist_y:.2f} meters, Z = {dist_z:.2f} meters")
else:
    print("No matching car found in both images.")
