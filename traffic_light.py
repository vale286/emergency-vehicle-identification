import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224  # Sesuai dengan ukuran model
MODEL_PATH = 'best_emergency_vehicle_model.keras'  # Path ke model
CATEGORIES = ['Emergency', 'Non-Emergency']  # Kategori

# Load model yang telah dilatih
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict_image(image_path):
    """
    Fungsi untuk memprediksi kategori gambar menggunakan model.
    :param image_path: Path ke gambar
    :return: label (Emergency/Non-Emergency), confidence (nilai prediksi)
    """
    if model is None:
        raise ValueError("Model is not loaded. Please check the model path.")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Resize dan normalisasi gambar
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Tambahkan batch dimension

    # Prediksi dengan model
    prediction = model.predict(img)[0][0]  # Hasil sigmoid (0 sampai 1)

    # Tentukan label berdasarkan threshold 0.5
    label = CATEGORIES[int(prediction > 0.5)]  # 0 -> Emergency, 1 -> Non-Emergency
    confidence = prediction if label == 'Emergency' else 1 - prediction

    return label, confidence

def traffic_light_status(image_path):
    """
    Fungsi untuk menentukan status lampu lalu lintas berdasarkan gambar kendaraan.
    :param image_path: Path ke gambar
    :return: Status lampu lalu lintas ("Green" atau "Red")
    """
    try:
        label, confidence = predict_image(image_path)
        print(f"Label: {label}, Confidence: {confidence:.2f}")  # Debug log

        # Atur status lampu berdasarkan prediksi
        if label == 'Emergency' and confidence > 0.02:  # Threshold confidence
            return "Green"  # Lampu hijau untuk kendaraan darurat
        return "Red"  # Lampu merah untuk kendaraan non-darurat
    except ValueError as e:
        print(f"Error: {e}")
        return "Error"
