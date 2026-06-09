import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter
from tqdm import tqdm

# Path ke dataset
DATASET_PATH = "D:/Emergency Vehicle Identification/dataset"
CATEGORIES = ["emergency", "non_emergency"]
IMG_SIZE = 224  # Ukuran gambar untuk model (disesuaikan)
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# Validasi dataset path dan kategori
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

for category in CATEGORIES:
    folder_path = os.path.join(DATASET_PATH, category)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Category folder not found: {folder_path}")

def load_images():
    images = []
    labels = []
    
    for category in CATEGORIES:
        folder_path = os.path.join(DATASET_PATH, category)
        label = CATEGORIES.index(category)

        for img_name in tqdm(os.listdir(folder_path), desc=f"Processing {category}"):
            if not img_name.lower().endswith(valid_extensions):
                continue
            try:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
                continue
    
    images = np.array(images) / 255.0  # Normalisasi (0-1)
    labels = np.array(labels)
    return images, labels

# Load dataset
images, labels = load_images()

# Cek jumlah label
label_counts = Counter(labels)
print("Label counts:", label_counts)

# Shuffle dataset
images, labels = shuffle(images, labels, random_state=42)

# Bagi dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print("Dataset berhasil diproses!")
print(f"Jumlah training data: {len(X_train)}, Jumlah testing data: {len(X_test)}")

# Simpan dataset yang telah di-preprocess
np.savez_compressed("dataset.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
