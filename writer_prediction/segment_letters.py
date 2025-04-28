import cv2
import os

def segment_letters(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letters = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100:  # noise filter
            letter = thresh[y:y+h, x:x+w]
            letter = cv2.resize(letter, (28, 28))
            letters.append((x, letter))

    letters.sort(key=lambda x: x[0])

    return [img for _, img in letters]

def save_letters_from_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for person in os.listdir(input_folder):
        person_folder = os.path.join(input_folder, person)
        if os.path.isdir(person_folder):
            for img_file in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_file)
                letters = segment_letters(img_path)
                for i, letter in enumerate(letters):
                    letter_output_folder = os.path.join(output_folder, person)
                    os.makedirs(letter_output_folder, exist_ok=True)
                    letter_path = os.path.join(letter_output_folder, f"{img_file.split('.')[0]}_letter_{i}.png")
                    cv2.imwrite(letter_path, letter)

if __name__ == "__main__":
    save_letters_from_folder('data/train', 'data/letters')
