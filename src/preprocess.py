import os
import cv2
import numpy as np

IMG_SIZE = 224

def load_dataset(data_dir):

    X = []
    y = []

    classes = os.listdir(data_dir)

    for label, activity in enumerate(classes):

        folder = os.path.join(data_dir, activity)

        for img_name in os.listdir(folder):

            img_path = os.path.join(folder, img_name)

            img = cv2.imread(img_path)

            img = cv2.resize(img,(IMG_SIZE, IMG_SIZE))

            img = img / 255.0

            X.append(img)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y