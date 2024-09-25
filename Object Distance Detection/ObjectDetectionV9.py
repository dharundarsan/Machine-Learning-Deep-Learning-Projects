from ultralytics import YOLO
import cv2
import numpy as np

# Distance calculation function
def find_dist(pt1, pt2, baseline, f_pixel):
    x1, y1 = pt1
    x2, y2 = pt2
    
    disparity = x1 - x2  # Difference in x-coordinates
    if disparity == 0:
        print("Disparity is zero, cannot compute distance")
        return 0.0, 0.0, 0.0  # Return zeros for distances if disparity is zero
    
    dist_z = (baseline * f_pixel) / disparity
    dist_x = (x1 - (640 // 2)) * dist_z / f_pixel  # Assuming image width is 640 pixels
    dist_y = y1 * dist_z / f_pixel
    
    return dist_x, dist_y, dist_z

# Object detection function with stereo image distance calculation
def detect_objects_in_stereo_images(left_image_path, right_image_path, baseline, focal_length, fov):
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")  # Specify your model if different

    # Load left and right images
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    # Perform object detection on the left image
    left_results = model(left_image_path)
    print(f"Results for Left Image ({left_image_path}):")
    left_boxes = []
    for result in left_results:
        for box in result.boxes:
            cls = box.cls.item()  # Convert to a scalar
            conf = box.conf.item()  # Convert to a scalar
            xyxy = box.xyxy.cpu().numpy().flatten()  # Flatten the array to handle single coordinates
            left_boxes.append({'class': cls, 'coordinates': xyxy, 'center': [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]})
            print(f"Class: {cls}, Confidence: {conf:.2f}, Coordinates: {xyxy}")

    # Perform object detection on the right image
    right_results = model(right_image_path)
    print(f"\nResults for Right Image ({right_image_path}):")
    right_boxes = []
    for result in right_results:
        for box in result.boxes:
            cls = box.cls.item()  # Convert to a scalar
            conf = box.conf.item()  # Convert to a scalar
            xyxy = box.xyxy.cpu().numpy().flatten()  # Flatten the array to handle single coordinates
            right_boxes.append({'class': cls, 'coordinates': xyxy, 'center': [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]})
            print(f"Class: {cls}, Confidence: {conf:.2f}, Coordinates: {xyxy}")

    # Find matching objects between left and right images
    if not left_boxes or not right_boxes:
        print("No matching objects found in both images.")
        return

    matched_objects = set()  # To track already matched objects

    for left_obj in left_boxes:
        for right_obj in right_boxes:
            # Match objects based on class and proximity (same class and similar y-coordinate)
            if left_obj['class'] == right_obj['class']:
                if np.abs(left_obj['center'][1] - right_obj['center'][1]) < 50:  # Compare y-coordinates for proximity
                    left_center = left_obj['center']
                    right_center = right_obj['center']

                    # Create a unique identifier for the matched object
                    obj_identifier = (left_obj['class'], tuple(left_obj['coordinates']))

                    if obj_identifier not in matched_objects:
                        matched_objects.add(obj_identifier)  # Add to the set to avoid reprinting
                        
                        # Calculate focal length in pixels
                        f_pixel = (640 * 0.5) / (np.tan(fov * 0.5 * np.pi / 180))  # Assuming image width is 640 pixels

                        # Calculate distance
                        dist_x, dist_y, dist_z = find_dist(left_center, right_center, baseline, f_pixel)
                        print(f"\nMatching object found: Class {left_obj['class']} (Common Car)")
                        print(f"Left Image Center: {left_center}, Right Image Center: {right_center}")
                        print(f"Distance: X={dist_x:.2f}m, Y={dist_y:.2f}m, Z={dist_z:.2f}m")

    # Optionally, visualize the results for both images
    left_results[0].show()  # Show the left image with bounding boxes
    right_results[0].show()  # Show the right image with bounding boxes

# Example usage
left_image_path = "D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\left2.jpg"
right_image_path = "D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\right2.jpg"

# Stereo camera parameters
baseline = 0.064  # Distance between the two cameras (in meters)
focal_length = 3.6  # Focal length of the camera lens (in mm)
fov = 90  # Field of view (in degrees)

detect_objects_in_stereo_images(left_image_path, right_image_path, baseline, focal_length, fov)
