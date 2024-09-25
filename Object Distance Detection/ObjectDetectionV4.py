import cv2
import numpy as np

# Update these paths to point to your downloaded files
weights_path = 'D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\yolov8.pt'  # Update the path if necessary
image_path = 'D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\left1.jpg'

# Load the YOLOv8 model
net = cv2.dnn.readNet(weights_path)

# Load the image
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Prepare the image for the network
blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)

# Run inference
output_layer_names = net.getUnconnectedOutLayersNames()
outputs = net.forward(output_layer_names)

# Process the outputs
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]  # Get scores for each class
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Threshold for detection
            # Get bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-max suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Load COCO classes (or any other classes you are using)
with open('D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Draw boxes and print details
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label + " " + str(confidences[i]), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
        # Print center coordinates and dimensions
        print(f"Class: {label}, Center: ({x + w // 2}, {y + h // 2}), Width: {w}, Height: {h}")

# Show the image with detections
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
