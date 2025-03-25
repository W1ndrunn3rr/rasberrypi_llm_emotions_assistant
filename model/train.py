import os 
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

def normalize_img(img):
    return (img - 127.5) / 127.5

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = tf.keras.applications.MobileNetV3Large(
    input_shape=(224,224,3),
    include_top=True,
    weights=None,
    classes=7,
    classifier_activation='softmax',
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)
model.summary()

# Data loading and preprocessing

data_dir = "images"

train_dir = os.path.join(data_dir, "train")
validation_dir = os.path.join(data_dir,"val")
test_dir = os.path.join(data_dir, "test")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
   preprocessing_function=normalize_img,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
 
)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=normalize_img)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=normalize_img)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224,),
    batch_size=64,
    class_mode='categorical',
    color_mode='rgb'
)

test_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    color_mode='rgb'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    color_mode='rgb'
)

class_names = list(train_generator.class_indices.keys())
num_classes = train_generator.num_classes

unique, counts = np.unique(train_generator.labels, return_counts=True)
class_counts = dict(zip(class_names, counts))

print(f"----------\nŁączna liczba klas: {num_classes}")
print("Liczba przykładów w każdej klasie:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")
print("---------\n")

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping, lr_scheduler]
)

converter = tf.lite.TFLiteConverter.from_saved_model('fer')  # Ścieżka do SavedModel
tflite_model = converter.convert()

# Zapisz model TFLite do pliku
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
    
class_names = list(train_generator.class_indices.keys())
val_images, val_labels = next(validation_generator)
predictions = model.predict(val_images)
y_true = np.argmax(val_labels, axis=1)
y_pred = np.argmax(predictions, axis=1)


print(classification_report(y_true, y_pred, target_names=class_names))