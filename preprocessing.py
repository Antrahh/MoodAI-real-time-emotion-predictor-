import os
import cv2
import numpy as np

categories = ["happy", "sad", "angry","neutral"]
img_size = 48

data = []
labels = []

for idx, category in enumerate(categories):
    cat_path = os.path.join(extract_path, category)
    for root, dirs, files in os.walk(cat_path):  # recursive
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                    data.append(img)
                    labels.append(idx)

data = np.array(data, dtype='float32') / 255.0
data = np.expand_dims(data, -1)
labels = np.array(labels)

print("Dataset ready!")
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)