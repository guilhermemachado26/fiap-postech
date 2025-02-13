import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_folder = "../dataset/dangerous"
output_folder = "../dataset/augmented"

os.makedirs(output_folder, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest",
)

for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)

    aug_count = 0
    for batch in datagen.flow(
        img_array,
        batch_size=1,
        save_to_dir=output_folder,
        save_prefix="aug",
        save_format="jpg",
    ):
        aug_count += 1
        if aug_count >= 10:
            break
