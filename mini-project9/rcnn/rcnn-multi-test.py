import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# === Class label mapping ===
class_map = {0: "mugs", 1: "computers", 2: "phones"}

# === Load trained model ===
model = tf.keras.models.load_model("multi_class_model.h5")

# === Load image ===
image_path = "./dataset_mini9/computers/3932d320d26c51d0d9d400c63c0b11ba4ab6688f.jpg"
image = cv2.imread(image_path)

cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

# ✅ Use high-quality mode to get larger, object-sized proposals
ss.switchToSelectiveSearchQuality()
ssresults = ss.process()

# === Track the best overall detection ===
best_score = 0
best_box = None
best_class_id = None

for e, result in enumerate(ssresults):
    if e >= 1000:
        break

    x, y, w, h = result
    if w < 50 or h < 50:  # Skip small boxes
        continue

    timage = image[y:y + h, x:x + w]
    if timage.shape[0] == 0 or timage.shape[1] == 0:
        continue

    resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
    resized = np.expand_dims(resized, axis=0)
    pred = model.predict(resized, verbose=0)
    score = np.max(pred)
    class_id = np.argmax(pred)

    if score > best_score:
        best_score = score
        best_box = [x, y, x + w, y + h]
        best_class_id = class_id

# === Annotate image ===
if best_box and best_score > 0.6:
    detected_class = class_map[best_class_id]
    print(f"✅ Detected class: {detected_class}")

    x1, y1, x2, y2 = best_box
    label_text = f"{detected_class} ({best_score:.2f})"

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label background
    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (0, 255, 0), cv2.FILLED)

    # Draw label text
    cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2)

# Save the image
cv2.imwrite("output_detected_computer.jpg", image)
print("✅ Saved result to output_detected_com.jpg")

# Optional: also display the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected Object")
plt.show()

