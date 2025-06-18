import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import urllib.request
import pytesseract

# Configure Tesseract path for Windows
try:
    import pytesseract
    # Try to set the Tesseract path for Windows
    if os.name == 'nt':  # Windows
        # Common Tesseract installation paths on Windows
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"Tesseract found at: {path}")
                break
        else:
            print("Warning: Tesseract not found in common locations. Make sure it's in your PATH.")
except ImportError:
    print("Warning: pytesseract not installed. OCR functionality will be limited.")

# Custom model import
try:
    from inferenceModel import ImageToWordModel
    from mltu.configs import BaseModelConfigs
except ImportError:
    ImageToWordModel = None
    BaseModelConfigs = None

class TextRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Recognition (Tesseract or Custom Model)")
        self.root.geometry("900x700")
        self.current_image = None
        self.current_image_path = None
        self.config_path = ""
        self.recognition_method = tk.StringVar(value="Tesseract")
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        title_label = ttk.Label(main_frame, text="Text Recognition", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Recognition method selection
        ttk.Label(main_frame, text="Recognition Method:").grid(row=1, column=0, sticky=tk.W, pady=5)
        method_combo = ttk.Combobox(main_frame, textvariable=self.recognition_method, values=["Tesseract", "Custom Model"], state="readonly", width=20)
        method_combo.grid(row=1, column=1, sticky=tk.W, padx=(5, 5), pady=5)
        method_combo.bind("<<ComboboxSelected>>", self.on_method_change)

        # Config path for custom model
        self.config_var = tk.StringVar()
        self.config_entry = ttk.Entry(main_frame, textvariable=self.config_var, width=50, state="disabled")
        self.config_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        self.config_browse_btn = ttk.Button(main_frame, text="Browse Config", command=self.browse_config, state="disabled")
        self.config_browse_btn.grid(row=2, column=2, pady=5)
        ttk.Label(main_frame, text="Custom Model Config:").grid(row=2, column=0, sticky=tk.W, pady=5)

        # Image selection
        ttk.Label(main_frame, text="Image:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.image_var = tk.StringVar()
        image_entry = ttk.Entry(main_frame, textvariable=self.image_var, width=60)
        image_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_image).grid(row=3, column=2, pady=5)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=20)
        ttk.Button(button_frame, text="Recognize Text", command=self.recognize_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_results).pack(side=tk.LEFT, padx=5)

        # Image display
        image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        image_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        self.image_label = ttk.Label(image_frame, text="No image loaded")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Results
        results_frame = ttk.LabelFrame(main_frame, text="Recognition Results", padding="10")
        results_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        results_frame.columnconfigure(0, weight=1)
        self.result_text = tk.Text(results_frame, height=6, wrap=tk.WORD)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

    def on_method_change(self, event=None):
        method = self.recognition_method.get()
        if method == "Custom Model":
            self.config_entry.config(state="normal")
            self.config_browse_btn.config(state="normal")
        else:
            self.config_entry.config(state="disabled")
            self.config_browse_btn.config(state="disabled")

    def browse_config(self):
        filename = filedialog.askopenfilename(
            title="Select Model Config File",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            self.config_var.set(filename)
            self.config_path = filename

    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
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

    def recognize_text(self):
        method = self.recognition_method.get()
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            messagebox.showerror("Error", "Please select a valid image file")
            return
        self.result_text.delete(1.0, tk.END)
        if method == "Tesseract":
            self.status_var.set("Running Tesseract OCR on whole image...")
            self.root.update()
            try:
                image = cv2.imread(self.current_image_path)
                if image is None:
                    raise ValueError("Could not load image")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config='--psm 6')
                self.result_text.insert(tk.END, f"Tesseract OCR Result:\n{'-'*30}\n{text.strip()}\n")
                self.status_var.set("Tesseract OCR completed")
            except Exception as e:
                self.status_var.set("Error during Tesseract OCR")
                messagebox.showerror("Error", f"Error during Tesseract OCR: {e}")
        elif method == "Custom Model":
            if not self.config_var.get() or not os.path.exists(self.config_var.get()):
                messagebox.showerror("Error", "Please select a valid custom model config file")
                return
            if ImageToWordModel is None or BaseModelConfigs is None:
                messagebox.showerror("Error", "Custom model code not available.")
                return
            self.status_var.set("Running custom model inference...")
            self.root.update()
            try:
                configs = BaseModelConfigs.load(self.config_var.get())
                model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
                image = cv2.imread(self.current_image_path)
                if image is None:
                    raise ValueError("Could not load image")
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                text = model.predict(image)
                self.result_text.insert(tk.END, f"Custom Model OCR Result:\n{'-'*30}\n{text.strip()}\n")
                self.status_var.set("Custom model inference completed")
            except Exception as e:
                self.status_var.set("Error during custom model inference")
                messagebox.showerror("Error", f"Error during custom model inference: {e}")

    def save_results(self):
        """Save the recognition result text to a .txt file."""
        result = self.result_text.get(1.0, tk.END).strip()
        if not result:
            messagebox.showwarning("Warning", "No recognition result to save.")
            return
        default_name = "result.txt"
        if self.current_image_path:
            base = os.path.splitext(os.path.basename(self.current_image_path))[0]
            default_name = f"{base}_result.txt"
        filename = filedialog.asksaveasfilename(
            title="Save Recognition Result",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(result)
                messagebox.showinfo("Success", f"Recognition result saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save result: {e}")

    def clear_results(self):
        self.result_text.delete(1.0, tk.END)
        self.image_label.configure(image="", text="No image loaded")
        self.image_label.image = None
        self.current_image_path = None
        self.image_var.set("")
        self.status_var.set("Ready")
        self.config_var.set("")
        self.config_path = ""
        self.recognition_method.set("Tesseract")
        self.config_entry.config(state="disabled")
        self.config_browse_btn.config(state="disabled")

def main():
    root = tk.Tk()
    app = TextRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 