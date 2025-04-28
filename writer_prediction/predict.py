import numpy as np
import tensorflow as tf
from segment_letters import segment_letters
from collections import Counter

def predict_writer(image_path):
    model = tf.keras.models.load_model('writer_model.h5')
    label_classes = np.load('label_classes.npy')

    letters = segment_letters(image_path)

    preds = []
    for letter in letters:
        letter = letter.reshape(1,28,28,1).astype('float32') / 255.0
        pred = model.predict(letter)
        pred_label = label_classes[np.argmax(pred)]
        preds.append(pred_label)

    # Majority voting
    final_prediction = Counter(preds).most_common(1)[0][0]
    return final_prediction

if __name__ == "__main__":
    image_path = 'data/test/unknown1.png'
    predicted_writer = predict_writer(image_path)
    print(f"Predicted writer: {predicted_writer}")
