import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

# ✅ Define class-to-label map
class_map = {
    "mugs": 0,
    "computers": 1,
    "phones": 2
}

# Lists to store data
train_images = []
train_labels = []
svm_images = []
svm_labels = []

# IoU calculation
def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return max(0.0, min(1.0, iou))

# Selective Search setup
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# ✅ Iterate through each class
for class_name, class_id in class_map.items():
    image_dir = f"./dataset_mini9/{class_name}"
    label_dir = f"./dataset_mini9/{class_name}_annotation_csv"

    for i in os.listdir(label_dir):
        try:
            if i.endswith(".csv"):
                filename = i.replace(".csv", ".jpg")
                print(f"Processing {filename} in class {class_name}")
                image = cv2.imread(os.path.join(image_dir, filename))
                if image is None:
                    print(f"Failed to load image: {filename}")
                    continue
                df = pd.read_csv(os.path.join(label_dir, i))
                gtvalues = []

                for _, row in df.iterrows():
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})

                    timage = image[y1:y2, x1:x2]
                    resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                    svm_images.append(resized)
                    svm_labels.append(tf.keras.utils.to_categorical(class_id, num_classes=len(class_map)))

                ss.setBaseImage(image)
                ss.switchToSelectiveSearchFast()
                ssresults = ss.process()
                imout = image.copy()
                counter = 0
                falsecounter = 0
                flag = False

                for e, result in enumerate(ssresults):
                    if e < 2000 and not flag:
                        x, y, w, h = result
                        for gtval in gtvalues:
                            iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                            timage = imout[y:y + h, x:x + w]
                            if counter < 30 and iou > 0.7:
                                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(class_id)
                                counter += 1
                            elif falsecounter < 30 and iou < 0.3:
                                # Skip background label -1 to avoid training error
                                falsecounter += 1
                        if counter >= 30 and falsecounter >= 30:
                            flag = True
        except Exception as e:
            print(f"Error processing {i}: {e}")
            continue

# Convert to arrays
X_new = np.array(train_images)
Y_new = np.array(train_labels)

# VGG-based model setup
vgg = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')
for layer in vgg.layers[:-2]:
    layer.trainable = False
x = vgg.get_layer('fc2').output
x = Dense(len(class_map), activation='softmax')(x)
model = Model(vgg.input, x)
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['acc'])
model.summary()
model.fit(X_new, Y_new, batch_size=16, epochs=2, verbose=1, validation_split=0.05, shuffle=True)

# Plot training loss
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss", "Validation Loss"])
plt.savefig('chart_loss.png')
plt.show()

# Save model
model.save('multi_class_model.h5')
