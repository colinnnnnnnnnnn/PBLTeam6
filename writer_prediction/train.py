import os
import numpy as np
from keras import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from build_model import build_writer_classifier

def load_data(data_dir):
    images = []
    labels = []
    for writer in os.listdir(data_dir):
        writer_folder = os.path.join(data_dir, writer)
        if os.path.isdir(writer_folder):
            for img_file in os.listdir(writer_folder):
                img_path = os.path.join(writer_folder, img_file)
                img = load_img(img_path, color_mode='grayscale', target_size=(28,28))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(writer)
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    images, labels = load_data('data/letters')

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    model = build_writer_classifier(num_classes=len(label_encoder.classes_))

    model.fit(images, labels_encoded, epochs=10, batch_size=32, validation_split=0.2)

    model.save('writer_model.h5')

    # Save label encoder classes
    np.save('label_classes.npy', label_encoder.classes_)
