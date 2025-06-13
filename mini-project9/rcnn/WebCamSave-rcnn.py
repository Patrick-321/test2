import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque

# === Class label mapping ===
class_map = {0: "mugs", 1: "computers", 2: "phones", 3: "background"}

# === Load model ===
print("Loading model...")
model = tf.keras.models.load_model("multi_class_model.h5")
print("Model loaded successfully!")

# === Configuration ===
CONFIDENCE_THRESHOLD = 0.85
NMS_THRESHOLD = 0.3
FRAME_SKIP = 3  # Process every 3rd frame for better performance
DETECTION_SIZE = (224, 224)
WINDOW_SIZES = [(120, 120), (180, 180), (240, 240)]
STRIDE_RATIO = 0.3

# === FPS tracking ===
fps_times = deque(maxlen=30)


def non_max_suppression(boxes, scores, classes, threshold=0.3):
    """Simple NMS implementation"""
    if len(boxes) == 0:
        return [], [], []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores)
    classes = np.array(classes)

    # Sort by confidence
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]

        # Calculate intersection
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Calculate union
        area1 = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area2 = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = area1 + area2 - intersection

        # Calculate IoU and filter
        iou = intersection / (union + 1e-6)
        indices = indices[1:][iou <= threshold]

    return boxes[keep], scores[keep], classes[keep]


def sliding_window_detection(frame, model):
    """Simplified sliding window detection"""
    h, w = frame.shape[:2]
    all_boxes, all_scores, all_classes = [], [], []

    # Preprocess frame
    frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)

    for win_size in WINDOW_SIZES:
        win_h, win_w = win_size
        step = int(win_h * STRIDE_RATIO)

        windows = []
        coords = []

        # Generate windows
        for y in range(0, h - win_h + 1, step):
            for x in range(0, w - win_w + 1, step):
                window = frame_blur[y:y + win_h, x:x + win_w]

                # Skip uniform regions
                if np.std(window) > 15:
                    resized = cv2.resize(window, DETECTION_SIZE)
                    windows.append(resized)
                    coords.append([x, y, x + win_w, y + win_h])

        # Batch prediction
        if windows:
            batch = np.array(windows)
            batch = tf.keras.applications.vgg16.preprocess_input(batch)
            predictions = model.predict(batch, batch_size=16, verbose=0)

            for i, pred in enumerate(predictions):
                class_id = np.argmax(pred)
                confidence = pred[class_id]

                if class_id != 3 and confidence > CONFIDENCE_THRESHOLD:
                    all_boxes.append(coords[i])
                    all_scores.append(confidence)
                    all_classes.append(class_id)

    return all_boxes, all_scores, all_classes


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    last_detections = ([], [], [])
    last_process_time = time.time()

    print("Starting detection... Press 'q' to quit, 's' to save screenshot")

    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()

        # Process detection every N frames
        if frame_count % FRAME_SKIP == 0:
            process_start = time.time()
            boxes, scores, classes = sliding_window_detection(frame, model)

            # Apply NMS
            if boxes:
                boxes, scores, classes = non_max_suppression(boxes, scores, classes, NMS_THRESHOLD)
                last_detections = (boxes, scores, classes)

            process_time = time.time() - process_start
            last_process_time = process_time

        # Draw last known detections
        boxes, scores, classes = last_detections
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            score = scores[i]
            class_id = int(classes[i])

            label = class_map.get(class_id, "unknown")
            text = f"{label} {score:.2f}"

            # Draw detection
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(display_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Calculate correct FPS (display FPS, not processing FPS)
        loop_time = time.time() - loop_start
        fps_times.append(loop_time)
        display_fps = 1.0 / np.mean(fps_times) if fps_times else 0

        # Show info
        info_text = f"Display FPS: {display_fps:.1f} | Objects: {len(boxes)} | Process Time: {last_process_time:.2f}s"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Object Detection", display_frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"detection_{frame_count}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")


if __name__ == "__main__":
    main()