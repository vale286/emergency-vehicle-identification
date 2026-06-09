# Emergency Vehicle Identification & Traffic Light System

Proyek *Computer Vision* berbasis web menggunakan **Streamlit** dan **TensorFlow/Keras** untuk mendeteksi kendaraan darurat (Ambulans/Pemadam Kebakaran) secara otomatis guna mengontrol prioritas lampu lalu lintas.

---

## Fitur dan Teknologi
* **AI Detection:** Mengklasifikasikan gambar ke dalam kategori *Emergency* atau *Non-Emergency*.
* **Smart Traffic Light:** Simulasi otomatis mengubah lampu menjadi **Hijau** jika mendeteksi kendaraan darurat, dan **Merah** jika kendaraan biasa.
* **Tech Stack:** Python, Streamlit, OpenCV, TensorFlow/Keras.

---

## Cara Menjalankan Proyek

### 1. Persiapan Environment
Pastikan Anda sudah mengunduh proyek ini dan masuk ke direktori utama, lalu install *library* yang dibutuhkan:
```bash
pip install streamlit opencv-python tensorflow numpy Pillow
