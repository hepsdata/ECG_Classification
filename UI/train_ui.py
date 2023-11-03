import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
from tempfile import NamedTemporaryFile

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def save_figure(fig):
    tmpfile = NamedTemporaryFile(delete=False, suffix='.png')
    fig.savefig(tmpfile.name)
    return tmpfile.name


def save_dataframe_as_csv(df):
    tmpfile = NamedTemporaryFile(delete=False, suffix='.csv')
    df.to_csv(tmpfile.name, index=False)
    return tmpfile.name


# Disable the pyplot global warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# UI setup
st.title("Model Trainer")
model_choice = st.selectbox("Choose a model:", ["VGG16", "ResNet50", "DenseNet121"])
data_dir = st.text_input("Enter the dataset directory:")
data = pd.read_csv('training2017/REFERENCE-original.csv', header=None, names=['ID', 'Label'])
label_set = set(data['Label'])

epochs = st.number_input("Number of epochs:", min_value=1, value=5)


model_choices = ["VGG16", "ResNet50", "DenseNet121"]
selected_models = st.multiselect("Choose models:", model_choices, default=model_choices)

# 성능 데이터를 저장할 빈 DataFrame 생성
performance_data = pd.DataFrame(columns=['Model', 'Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss'])

if st.button("Train Model") and data_dir:
    for model_choice in selected_models:

        # Set parameters
        batch_size = 32
        img_height = 600
        img_width = 600

        # Load data
        train_ds = image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

        val_ds = image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )


        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Select model and preprocessing function
        if model_choice == "VGG16":
            base_model = VGG16(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
            preprocess = vgg_preprocess
        elif model_choice == "ResNet50":
            base_model = ResNet50(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
            preprocess = resnet_preprocess
        else:
            base_model = DenseNet121(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
            preprocess = densenet_preprocess

        train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
        val_ds = val_ds.map(lambda x, y: (preprocess(x), y))

        base_model.trainable = False

        data_augmentation = Sequential([
            layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
            layers.RandomRotation(0.1)
        ])

        # Determine the number of classes
        num_classes = 4

        model = Sequential([
            data_augmentation,
            base_model,
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        # Train the model
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        st.write(f"{model_choice} training completed!")

        # Accuracy & Loss visualization
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        avg_train_acc = np.mean(history.history['accuracy'])
        avg_val_acc = np.mean(history.history['val_accuracy'])
        avg_train_loss = np.mean(history.history['loss'])
        avg_val_loss = np.mean(history.history['val_loss'])

        performance_data = performance_data.append({
            'Model': model_choice,
            'Training Accuracy': avg_train_acc,
            'Validation Accuracy': avg_val_acc,
            'Training Loss': avg_train_loss,
            'Validation Loss': avg_val_loss
        }, ignore_index=True)
        st.table(performance_data)

        saved_csv_path = save_dataframe_as_csv(performance_data)
        st.markdown(f"[Click here to download the data as CSV]({saved_csv_path})")


        epochs_range = range(epochs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.plot(epochs_range, acc, label='Training Accuracy')
        ax1.plot(epochs_range, val_acc, label='Validation Accuracy')
        ax1.legend(loc='lower right')
        ax1.set_title('Training and Validation Accuracy')

        ax2.plot(epochs_range, loss, label='Training Loss')
        ax2.plot(epochs_range, val_loss, label='Validation Loss')
        ax2.legend(loc='upper right')
        ax2.set_title('Training and Validation Loss')
        st.pyplot(fig)
        saved_path = save_figure(fig)
        st.markdown(f"[Click here to download the figure]({saved_path})")

        y_true = []
        for images, labels in val_ds:
            y_true.extend(labels.numpy())

        # 예측값 (y_pred) 생성
        y_pred = []
        for images, labels in val_ds:
            preds = model.predict(images)
            preds = np.argmax(preds, axis=1)
            y_pred.extend(preds)

        # 오차 행렬 생성 및 시각화
        matrix = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10,7))
        sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt='g', ax=ax, xticklabels=label_set, yticklabels=label_set)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)


