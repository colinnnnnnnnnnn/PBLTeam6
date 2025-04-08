import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import os

# Load the dataset (same as previous script)
def load_dataset(csv_path):
    """
    Load the dataset from CSV and prepare images and labels

    Args:
        csv_path (str): Path to the CSV file containing image paths and labels

    Returns:
        tuple: X (images), y (labels)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Prepare lists to store images and labels
    images = []
    labels = []

    # Process each image
    for _, row in df.iterrows():
        # Read image
        img_path = row['image']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize image to a consistent size (e.g., 28x28)
        img = cv2.resize(img, (28, 28))

        # Normalize pixel values
        img = img / 255.0

        images.append(img)
        labels.append(row['label'])

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Reshape images for TensorFlow (add channel dimension)
    X = X.reshape(-1, 28, 28, 1)

    return X, y


# Create the neural network model (same as previous script)
def create_model(input_shape, num_classes):
    """
    Create a Convolutional Neural Network for character recognition

    Args:
        input_shape (tuple): Shape of input images
        num_classes (int): Number of unique characters/classes

    Returns:
        tf.keras.Model: Compiled neural network model
    """
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Train and save the model
def train_and_save_model(csv_path, model_path='handwritten_character_model.h5',
                         label_encoder_path='label_encoder.pkl', epochs=100, batch_size=32):
    """
    Train the model and save it along with the label encoder

    Args:
        csv_path (str): Path to the CSV file with image paths and labels
        model_path (str): Path to save the trained model
        label_encoder_path (str): Path to save the label encoder
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training

    Returns:
        tuple: Trained model and label encoder
    """
    # Validate CSV file
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load dataset
    X, y = load_dataset(csv_path)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Get number of classes
    num_classes = len(np.unique(y_encoded))

    # Create and train the model
    model = create_model((28, 28, 1), num_classes)

    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_split=0.2,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    # Detailed evaluation
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

    # Save the model
    model.save(model_path)

    # Save the label encoder
    import joblib
    joblib.dump(label_encoder, label_encoder_path)

    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {label_encoder_path}")

    return model, label_encoder


# Load the trained model and label encoder
def load_trained_model(model_path='handwritten_character_model.h5',
                       label_encoder_path='label_encoder.pkl'):
    """
    Load a previously trained model and its label encoder

    Args:
        model_path (str): Path to the saved model
        label_encoder_path (str): Path to the saved label encoder

    Returns:
        tuple: Loaded model and label encoder
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Load the label encoder
    import joblib
    label_encoder = joblib.load(label_encoder_path)

    return model, label_encoder


# Predict character for a single image
def predict_character(model, label_encoder, image_path):
    """
    Predict the character in a single image

    Args:
        model (tf.keras.Model): Trained model
        label_encoder (LabelEncoder): Label encoder
        image_path (str): Path to the image to predict

    Returns:
        str: Predicted character
    """
    # Read and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)

    # Decode the predicted label
    return label_encoder.inverse_transform(predicted_class)[0]


# Example usage
if __name__ == "__main__":
    csv_path = 'english.csv'
    model_path = 'handwritten_character_model.h5'
    label_encoder_path = 'label_encoder.pkl'

    # Option 1: Train and save the model (do this once)
    if not os.path.exists(model_path):
        train_and_save_model(csv_path)

    # Option 2: Load the trained model
    model, label_encoder = load_trained_model()

    # Make a prediction
    sample_image_path = 'Img/img017-009.png'
    predicted_char = predict_character(model, label_encoder, sample_image_path)
    print(f"Predicted character: {predicted_char}")
