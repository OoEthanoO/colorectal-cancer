import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.image import resize_with_pad
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import seaborn as sns
import json

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

images = np.load('images.npy')
labels = np.load('labels.npy')

labels_ohe = pd.get_dummies(labels)
label_names = labels_ohe.columns.tolist()

with open('label_names.json', 'w') as f:
    json.dump(label_names, f)

y = labels_ohe.values
X = images

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

def plot_metric(history, metric="accuracy", best_is_max=True, start_epoch=0, random_model_metric=None, model=None, X_test=None, y_test=None):
    training_accuracy = history.history[metric][start_epoch:]
    validation_accuracy = history.history['val_' + metric][start_epoch:]

    if best_is_max:
        best_epoch = validation_accuracy.index(max(validation_accuracy))
    else:
        best_epoch = validation_accuracy.index(min(validation_accuracy))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.title(metric.capitalize() + ' as Model Trains')
    plt.xlabel('Epoch #')
    plt.ylabel(metric.capitalize())

    plt.plot(training_accuracy, label='Train')
    plt.plot(validation_accuracy, label='Validation')
    plt.axvline(x=best_epoch, linestyle='--', color='green', label='Best Epoch')

    if random_model_metric is not None:
        plt.axhline(random_model_metric, linestyle='--', color='red', label='Chance')

    plt.legend()

    if model is not None and X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred_classes)

        plt.subplot(1, 2, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

    plt.show()

def ResizeImages(images, height, width):
    return np.array([resize_with_pad(image, height, width, antialias=True) for image in images]).astype(int)

mobile_net = VGG16(include_top=True)

new_output_layer = Dense(8, activation='softmax')

output = new_output_layer(mobile_net.layers[-2].output)
input = mobile_net.input
transfer_cnn = Model(input, output)

for layer in transfer_cnn.layers:
    layer.trainable = False

transfer_cnn.layers[-1].trainable = True

transfer_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])

transfer_cnn.summary()

X_train_resized = ResizeImages(X_train, 224, 224)
X_test_resized = ResizeImages(X_test, 224, 224)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train_resized)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
model_checkpoint = ModelCheckpoint('best_transfer_cnn_2.h5', monitor='val_accuracy', save_best_only=True, mode='max')

history_transfer = transfer_cnn.fit(datagen.flow(X_train_resized, y_train, batch_size=32),
                                    epochs=50,
                                    validation_data=(X_test_resized, y_test),
                                    callbacks=[early_stopping, reduce_lr, model_checkpoint])

plot_metric(history_transfer, model=transfer_cnn, X_test=X_test_resized, y_test=y_test)