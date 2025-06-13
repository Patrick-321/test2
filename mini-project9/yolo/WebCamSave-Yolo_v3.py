# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4 --device 'cpu'
#                 python3 object_detection_yolo.py --video=run.mp4 --device 'gpu'
#                 python3 object_detection_yolo.py --image=bird.jpg --device 'cpu'
#                 python3 object_detection_yolo.py --image=bird.jpg --device 'gpu'

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import time
from collections import deque

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--device', default='cpu', help="Device to perform inference on 'cpu' or 'gpu'.")
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Define target classes we want to detect
TARGET_CLASSES = ['laptop', 'cell phone', '']
TARGET_CLASS_IDS = []

# Find the class IDs for our target classes
print("🔍 Searching for target classes in COCO dataset...")
print("Available classes:", classes[:10], "... (showing first 10)")

for i, class_name in enumerate(classes):
    if class_name in TARGET_CLASSES:
        TARGET_CLASS_IDS.append(i)
        print(f"✅ Target class '{class_name}' found at ID: {i}")

# Debug: Show all target class mappings
print(f"\n🎯 Target Classes Configuration:")
for target_class in TARGET_CLASSES:
    found = False
    for i, class_name in enumerate(classes):
        if class_name == target_class:
            print(f"  - {target_class} → Class ID: {i}")
            found = True
            break
    if not found:
        print(f"  - {target_class} → ❌ NOT FOUND!")

if not TARGET_CLASS_IDS:
    print("❌ Warning: No target classes found in COCO names file!")
    print("Available classes:", classes)
    sys.exit(1)

print(f"✅ Successfully loaded {len(TARGET_CLASS_IDS)} target classes: {TARGET_CLASS_IDS}")

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

if (args.device == 'cpu'):
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    print('Using CPU device.')
elif (args.device == 'gpu'):
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print('Using GPU device.')

# Video recording variables
recording = False
record_start_time = 0
record_duration = 5  # 5 seconds
frame_buffer = deque(maxlen=150)  # Buffer to store frames (assuming 30 FPS)
group_counter = 1
temp_writer = None


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box with enhanced highlighting
def drawPred(classId, conf, left, top, right, bottom, frame):
    # Only draw if it's one of our target classes
    if classId not in TARGET_CLASS_IDS:
        return

    # Define colors and display names for each target class
    class_info = {
        'cup': {'color': (255, 0, 0), 'display_name': 'CUP'},  # Blue
        'cell phone': {'color': (0, 0, 255), 'display_name': 'CELL PHONE'},  # Red
        'keyboard': {'color': (0, 255, 0), 'display_name': 'KEYBOARD'}  # Green
    }

    class_name = classes[classId]
    color = class_info.get(class_name, {'color': (255, 178, 50), 'display_name': class_name.upper()})['color']
    display_name = class_info.get(class_name, {'color': (255, 178, 50), 'display_name': class_name.upper()})[
        'display_name']

    # Draw main bounding box with thicker border for better visibility
    cv.rectangle(frame, (left, top), (right, bottom), color, 4)

    # Add corner markers for enhanced visibility
    corner_length = 20
    corner_thickness = 6

    # Top-left corner
    cv.line(frame, (left, top), (left + corner_length, top), color, corner_thickness)
    cv.line(frame, (left, top), (left, top + corner_length), color, corner_thickness)

    # Top-right corner
    cv.line(frame, (right, top), (right - corner_length, top), color, corner_thickness)
    cv.line(frame, (right, top), (right, top + corner_length), color, corner_thickness)

    # Bottom-left corner
    cv.line(frame, (left, bottom), (left + corner_length, bottom), color, corner_thickness)
    cv.line(frame, (left, bottom), (left, bottom - corner_length), color, corner_thickness)

    # Bottom-right corner
    cv.line(frame, (right, bottom), (right - corner_length, bottom), color, corner_thickness)
    cv.line(frame, (right, bottom), (right, bottom - corner_length), color, corner_thickness)

    # Create label with object name and confidence
    confidence_text = f'{conf:.1%}'  # Show as percentage
    label = f'{display_name} ({confidence_text})'

    # Calculate label background size with padding
    font_scale = 0.8
    font_thickness = 2
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_DUPLEX, font_scale, font_thickness)

    # Position label above the bounding box, but ensure it's visible
    label_y = max(top - 10, labelSize[1] + 10)
    label_x = left

    # Ensure label doesn't go outside frame boundaries
    if label_x + labelSize[0] > frame.shape[1]:
        label_x = frame.shape[1] - labelSize[0] - 5
    if label_x < 0:
        label_x = 5

    # Draw label background with some transparency effect
    overlay = frame.copy()
    cv.rectangle(overlay,
                 (label_x - 5, label_y - labelSize[1] - 8),
                 (label_x + labelSize[0] + 5, label_y + baseLine + 5),
                 color, -1)

    # Blend the overlay with the original frame for transparency
    alpha = 0.8
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw border around label
    cv.rectangle(frame,
                 (label_x - 5, label_y - labelSize[1] - 8),
                 (label_x + labelSize[0] + 5, label_y + baseLine + 5),
                 (255, 255, 255), 2)

    # Draw the text label in white for contrast
    cv.putText(frame, label, (label_x, label_y),
               cv.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), font_thickness)

    # Add a subtle glow effect around the text
    cv.putText(frame, label, (label_x, label_y),
               cv.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), font_thickness + 1)


# Start recording function
def start_recording(frame_width, frame_height):
    global temp_writer, recording, record_start_time, group_counter

    if not recording:
        output_filename = f"yolo_group5.mp4"
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        temp_writer = cv.VideoWriter(output_filename, fourcc, 30.0, (int(frame_width), int(frame_height)))

        # Write buffered frames to the video
        for buffered_frame in frame_buffer:
            temp_writer.write(buffered_frame)

        recording = True
        record_start_time = time.time()
        print(f"🎬 Started recording: {output_filename}")
        group_counter += 1


# Stop recording function
def stop_recording():
    global temp_writer, recording

    if recording and temp_writer:
        temp_writer.release()
        temp_writer = None
        recording = False
        print("🎬 Recording stopped and saved!")


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    print(f"🔍 Processing frame: {frameWidth}x{frameHeight}")
    print(f"📊 Network outputs: {len(outs)} detection layers")

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    target_detected = False
    all_detections = 0
    target_detections = 0

    for layer_idx, out in enumerate(outs):
        print(f"  Layer {layer_idx}: {len(out)} detections")
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            all_detections += 1

            # Debug: Show detection info for high confidence detections
            if confidence > 0.3:  # Lower threshold for debugging
                class_name = classes[classId] if classId < len(classes) else "Unknown"
                print(f"    Detection: {class_name} (ID: {classId}) - Confidence: {confidence:.3f}")

            # Check if it's one of our target classes and above threshold
            if classId in TARGET_CLASS_IDS and confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                target_detected = True
                target_detections += 1

                class_name = classes[classId]
                print(
                    f"    ✅ TARGET DETECTED: {class_name} - Confidence: {confidence:.3f} - Box: ({left},{top},{width},{height})")

    print(f"📈 Total detections: {all_detections}, Target detections: {target_detections}")

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    if boxes:
        print(f"🎯 Applying NMS to {len(boxes)} target detections...")
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        print(f"📋 After NMS: {len(indices) if len(indices) > 0 else 0} final detections")

        if len(indices) > 0:
            for i in indices:
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                class_name = classes[classIds[i]]
                print(f"    🎨 Drawing: {class_name} at ({left},{top}) - {width}x{height}")
                drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)
    else:
        print("❌ No target objects detected in this frame")

    # Handle recording logic
    global recording, record_start_time

    if target_detected and not recording:
        start_recording(frameWidth, frameHeight)

    if recording:
        current_time = time.time()
        if current_time - record_start_time >= record_duration:
            stop_recording()
        elif temp_writer:
            temp_writer.write(frame)

    return target_detected


# Process inputs
winName = 'YOLO Object Detection - Cups, Cell Phones & Keyboards'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_yolo_out_py.mp4'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

print("\n🚀 YOLO Detection Started!")
print("🎯 Detecting: Cups, Cell Phones, Keyboards")
print("📹 Will automatically record 5-second clips when objects are detected!")
print("Press 'q' to quit\n")

frame_count = 0

while cv.waitKey(1) < 0:
    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        if recording:
            stop_recording()
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # Store frame in buffer for potential recording
    frame_buffer.append(frame.copy())

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    objects_detected = postprocess(frame, outs)

    # Put efficiency information
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Add recording status
    if recording:
        remaining_time = record_duration - (time.time() - record_start_time)
        record_label = f'🔴 Recording: {remaining_time:.1f}s left'
        cv.putText(frame, record_label, (0, 45), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Add detection debug info to the frame
    debug_info = [
        f'Confidence Threshold: {confThreshold}',
        f'Total Detections: {all_detections if "all_detections" in locals() else "Processing..."}',
        f'Target Detections: {target_detections if "target_detections" in locals() else "Processing..."}',
        f'Target Classes: {len(TARGET_CLASS_IDS)} loaded'
    ]

    for i, info in enumerate(debug_info):
        cv.putText(frame, info, (10, 80 + i * 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)

    # Check for 'q' key press to quit
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        if recording:
            stop_recording()
        break

# Cleanup
cap.release()
if vid_writer:
    vid_writer.release()
cv.destroyAllWindows()
print("🎉 Program finished successfully!")