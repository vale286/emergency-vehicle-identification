import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Normalisasi label
y_train = np.array(y_train).astype('float32')
y_test = np.array(y_test).astype('float32')

# Augmentasi data
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=8)

# Load MobileNetV2 tanpa bagian atas (top layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Tambahkan layer untuk klasifikasi
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Bangun model
model = Model(inputs=base_model.input, outputs=predictions)

# Bekukan layer pretrained untuk tahap awal
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_emergency_vehicle_model.keras", monitor='val_loss', save_best_only=True)

# Train model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, checkpoint]
)

# Fine-tuning: Buka sebagian layer pretrained
for layer in base_model.layers[-30:]:  # Buka 30 layer terakhir
    layer.trainable = True

# Re-compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune model
history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, checkpoint]
)

# Evaluasi
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Visualisasi
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy")
plt.show()
