import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import shutil
import json
from pathlib import Path
import sys

# --- OCR dependencies ---
try:
    import pytesseract
    # Tesseract path config for Windows
    if os.name == 'nt':
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
except ImportError:
    pytesseract = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None
try:
    from tensorflow import keras
except ImportError:
    keras = None

# --- Writer Identification dependencies ---
import importlib.util
writer_id_path = os.path.join(os.path.dirname(__file__), 'writer-identification', 'predict.py')
# from writer-identification.model import WriterIdentifier
WriterIdentifier = None
if os.path.exists(writer_id_path):
    spec = importlib.util.spec_from_file_location("writer_identification_predict", writer_id_path)
    writer_id_module = importlib.util.module_from_spec(spec)
    sys.modules["writer_identification_predict"] = writer_id_module
    try:
        spec.loader.exec_module(writer_id_module)
        WriterIdentifier = getattr(writer_id_module, "WriterIdentifier", None)
    except Exception as e:
        print("Error importing WriterIdentifier:", e)
        import traceback; traceback.print_exc()
        WriterIdentifier = None

class UnifiedHandwritingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Analysis: Writer ID & Text Recognition")
        self.root.geometry("1100x800")
        self.current_image_path = None
        self.writer_model_path = tk.StringVar()
        self.text_model_type = tk.StringVar(value="Tesseract")
        self.text_model_path = tk.StringVar()
        self.writer_result = None
        self.text_result = None
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # Title
        ttk.Label(main_frame, text="Handwriting Analysis", font=("Arial", 18, "bold")).grid(row=0, column=0, columnspan=4, pady=(0, 20))

        # Image selection
        ttk.Label(main_frame, text="Image:").grid(row=1, column=0, sticky=tk.W)
        self.image_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.image_var, width=60).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_image).grid(row=1, column=2, padx=5)
        ttk.Button(main_frame, text="Clear", command=self.clear_all).grid(row=1, column=3, padx=5)

        # Image preview
        image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        image_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        self.image_label = ttk.Label(image_frame, text="No image loaded")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Writer Identification Section ---
        writer_frame = ttk.LabelFrame(main_frame, text="Writer Identification", padding="10")
        writer_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N), pady=10)
        writer_frame.columnconfigure(1, weight=1)
        ttk.Label(writer_frame, text="Model (.pth):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(writer_frame, textvariable=self.writer_model_path, width=40).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(writer_frame, text="Browse", command=self.browse_writer_model).grid(row=0, column=2, padx=5)
        ttk.Button(writer_frame, text="Identify Writer", command=self.identify_writer).grid(row=1, column=0, columnspan=3, pady=10)
        self.writer_result_text = tk.Text(writer_frame, height=6, width=60, wrap=tk.WORD)
        self.writer_result_text.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E))

        # --- Text Recognition Section ---
        text_frame = ttk.LabelFrame(main_frame, text="Text Recognition", padding="10")
        text_frame.grid(row=3, column=2, columnspan=2, sticky=(tk.W, tk.E, tk.N), pady=10)
        text_frame.columnconfigure(1, weight=1)
        ttk.Label(text_frame, text="Method:").grid(row=0, column=0, sticky=tk.W)
        method_combo = ttk.Combobox(text_frame, textvariable=self.text_model_type, values=["Tesseract", "ONNX", "Keras (.h5)"], state="readonly", width=15)
        method_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        method_combo.bind("<<ComboboxSelected>>", self.on_text_method_change)
        ttk.Label(text_frame, text="Model (if needed):").grid(row=1, column=0, sticky=tk.W)
        self.text_model_entry = ttk.Entry(text_frame, textvariable=self.text_model_path, width=40, state="disabled")
        self.text_model_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        self.text_model_browse_btn = ttk.Button(text_frame, text="Browse", command=self.browse_text_model, state="disabled")
        self.text_model_browse_btn.grid(row=1, column=2, padx=5)
        ttk.Button(text_frame, text="Recognize Text", command=self.recognize_text).grid(row=2, column=0, columnspan=3, pady=10)
        self.text_result_text = tk.Text(text_frame, height=6, width=60, wrap=tk.WORD)
        self.text_result_text.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E))

        # --- Add New Author Section ---
        ttk.Button(main_frame, text="Add New Author", command=self.add_new_author).grid(row=4, column=0, columnspan=4, pady=10)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=5, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))

    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        if filename:
            self.image_var.set(filename)
            self.current_image_path = filename
            self.load_image_preview()

    def load_image_preview(self):
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            return
        try:
            image = Image.open(self.current_image_path)
            max_size = (400, 400)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def clear_all(self):
        self.image_var.set("")
        self.current_image_path = None
        self.image_label.configure(image="", text="No image loaded")
        self.writer_result_text.delete(1.0, tk.END)
        self.text_result_text.delete(1.0, tk.END)
        self.status_var.set("Ready")

    def browse_writer_model(self):
        filename = filedialog.askopenfilename(title="Select PyTorch Model (.pth)", filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")])
        if filename:
            self.writer_model_path.set(filename)

    def on_text_method_change(self, event=None):
        method = self.text_model_type.get()
        if method == "Tesseract":
            self.text_model_entry.config(state="disabled")
            self.text_model_browse_btn.config(state="disabled")
        else:
            self.text_model_entry.config(state="normal")
            self.text_model_browse_btn.config(state="normal")

    def browse_text_model(self):
        method = self.text_model_type.get()
        if method == "ONNX":
            filetypes = [("ONNX model", "*.onnx"), ("All files", "*.*")]
        else:
            filetypes = [("Keras model", "*.h5"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(title="Select Model File", filetypes=filetypes)
        if filename:
            self.text_model_path.set(filename)

    def identify_writer(self):
        self.writer_result_text.delete(1.0, tk.END)
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            messagebox.showerror("Error", "Please select a valid image file")
            return
        if not self.writer_model_path.get():
            messagebox.showerror("Error", "Please select a model file for writer identification")
            return
        if WriterIdentifier is None:
            messagebox.showerror("Error", "WriterIdentifier class not available. Check dependencies.")
            return
        self.status_var.set("Identifying writer...")
        self.root.update()
        try:
            model_path = Path(self.writer_model_path.get())
            identifier = WriterIdentifier(model_path, device="cpu")
            results = identifier.predict_image(Path(self.current_image_path), top_k=5)
            self.writer_result = results
            if not results:
                self.writer_result_text.insert(tk.END, "No prediction.")
            else:
                for pred in results:
                    self.writer_result_text.insert(tk.END, f"{pred['rank']}. {pred['writer_id']} (confidence: {pred['confidence']:.3f})\n")
            self.status_var.set("Writer identification completed.")
        except Exception as e:
            self.status_var.set("Error during writer identification")
            messagebox.showerror("Error", f"Error during writer identification: {e}")

    def recognize_text(self):
        self.text_result_text.delete(1.0, tk.END)
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            messagebox.showerror("Error", "Please select a valid image file")
            return
        method = self.text_model_type.get()
        self.status_var.set(f"Running {method} text recognition...")
        self.root.update()
        try:
            if method == "Tesseract":
                if pytesseract is None:
                    raise RuntimeError("pytesseract not installed")
                image = cv2.imread(self.current_image_path)
                if image is None:
                    raise ValueError("Could not load image")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config='--psm 6')
                self.text_result = text.strip()
                self.text_result_text.insert(tk.END, f"Tesseract OCR Result:\n{'-'*30}\n{text.strip()}\n")
            elif method == "ONNX":
                if ort is None:
                    raise RuntimeError("onnxruntime not installed")
                if not self.text_model_path.get():
                    raise RuntimeError("Please select an ONNX model file")
                image = cv2.imread(self.current_image_path)
                if image is None:
                    raise ValueError("Could not load image")
                # Preprocess: resize to model input, normalize, etc. (assume 128x32, 3ch, float32, 0-1)
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (128, 32))
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)
                session = ort.InferenceSession(self.text_model_path.get())
                input_name = session.get_inputs()[0].name
                output = session.run(None, {input_name: img})[0]
                # Decoding: just show argmax chars for demo (real model needs vocab)
                pred = np.argmax(output, axis=2)[0]
                text = ''.join([chr(c) for c in pred if c > 0 and c < 128])
                self.text_result = text
                self.text_result_text.insert(tk.END, f"ONNX Model Result:\n{'-'*30}\n{text}\n")
            elif method == "Keras (.h5)":
                if keras is None:
                    raise RuntimeError("TensorFlow/Keras not installed")
                if not self.text_model_path.get():
                    raise RuntimeError("Please select a Keras .h5 model file")
                image = cv2.imread(self.current_image_path)
                if image is None:
                    raise ValueError("Could not load image")
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (128, 32))
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)
                model = keras.models.load_model(self.text_model_path.get())
                output = model.predict(img)
                pred = np.argmax(output, axis=2)[0]
                text = ''.join([chr(c) for c in pred if c > 0 and c < 128])
                self.text_result = text
                self.text_result_text.insert(tk.END, f"Keras Model Result:\n{'-'*30}\n{text}\n")
            self.status_var.set(f"{method} text recognition completed.")
        except Exception as e:
            self.status_var.set(f"Error during {method} text recognition")
            messagebox.showerror("Error", f"Error during {method} text recognition: {e}")

    def add_new_author(self):
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            messagebox.showerror("Error", "Please select a valid image file")
            return
        label = simpledialog.askstring("Add New Author", "Enter new author label (folder name):")
        if not label:
            return
        data_dir = os.path.join(os.path.dirname(__file__), "writer-identification", "data", label)
        os.makedirs(data_dir, exist_ok=True)
        # Copy image to new author folder
        base = os.path.basename(self.current_image_path)
        dest = os.path.join(data_dir, base)
        shutil.copy(self.current_image_path, dest)
        messagebox.showinfo("Success", f"Image added to new author '{label}'. You can retrain the model to include this author.")
        self.status_var.set(f"Added new author: {label}")

def main():
    root = tk.Tk()
    app = UnifiedHandwritingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 