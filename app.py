import tkinter as tk
from tkinter import filedialog, Label, Button
from joblib import load
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from collections import Counter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Label mapping
label_map = {0: "Chickenpox", 1: "Measles", 2: "Monkeypox", 3: "Normal"}
remedies = {
    "Chickenpox": "Home Remedies:\n- Neem Leaves Paste.\n- Wear loose fitted and light coloured clothes.\n- Stay hydrated and rest well.",
    "Measles": "Home Remedies:\n- Stay hydrated with water and fruit juices.\n- Use a cool mist humidifier.\n- Take Vitamin A supplements as recommended.",
    "Monkeypox": "Home Remedies:\n- Isolate yourself to prevent spread.\n- Keep skin lesions clean and dry.\n- Use Turmeric and Neem Paste:",
    "Normal": "You're Normal! No special remedies needed. Keep up with a healthy lifestyle."
}

# Load KNN and SVM models
knn = load('knn_model.joblib')
svm = load('svm_model.joblib')

# Load AlexNet model
alexnet = models.alexnet(weights=None)
alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # Match RGB
alexnet.classifier[6] = nn.Linear(4096, len(label_map))  # Match output classes
alexnet.load_state_dict(torch.load('alex_net.pth', map_location=device))
alexnet = alexnet.to(device).eval()

# Image preprocessing
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Functions for prediction
def predict_knn(image_path):
    image = Image.open(image_path).convert('L')
    image = data_transform(image).numpy().flatten()
    return knn.predict([image])[0]

def predict_svm(image_path):
    image = Image.open(image_path).convert('L')
    image = data_transform(image).numpy().flatten()
    return svm.predict([image])[0]

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Grayscale(), 
        transforms.Normalize(mean=[0.485], std=[0.229])  # Adjust for grayscale normalization
    ])
    image = Image.open(image_path).convert('RGB')  # AlexNet expects RGB
    return preprocess(image).unsqueeze(0).to(device)

def predict_alexnet(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = alexnet(image)
    _, preds = torch.max(outputs, 1)
    return preds.item()

def majority_voting(image_path):
    knn_pred = predict_knn(image_path)
    svm_pred = predict_svm(image_path)
    alexnet_pred = predict_alexnet(image_path)
    predictions = [knn_pred, svm_pred, alexnet_pred]
    most_common = Counter(predictions).most_common(1)[0][0]
    return label_map[most_common]

# Tkinter GUI
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        label_image.config(image=img_tk)
        label_image.image = img_tk

        # Get prediction
        prediction = majority_voting(file_path)
        label_result.config(text=f"Predicted Class: {prediction}")
        label_remedies.config(text=remedies[prediction])

# Initialize Tkinter
root = tk.Tk()
root.title("DermaScan: Skin Disease Recognition System")

# Full-screen settings
def toggle_fullscreen(event=None):
    root.attributes("-fullscreen", not root.attributes("-fullscreen"))

def exit_fullscreen(event=None):
    root.attributes("-fullscreen", False)

root.bind("<F11>", toggle_fullscreen)
root.bind("<Escape>", exit_fullscreen)

# Style settings
root.configure(bg="#f0f8ff")  # Light blue background
font_title = ("Helvetica", 18, "bold")
font_normal = ("Helvetica", 14)
font_result = ("Helvetica", 16, "bold")
font_remedies = ("Helvetica", 12)

# Create and place widgets
label_title = Label(root, text="DermaScan: ML-Driven Skin Disease Recognition System", font=font_title, bg="#f0f8ff", fg="#333")
label_title.pack(pady=20)

label_image = Label(root, bg="#f0f8ff")
label_image.pack()

btn_select = Button(root, text="Select Image", command=open_file, font=font_normal, bg="#4CAF50", fg="white", relief="raised")
btn_select.pack(pady=15)

label_result = Label(root, text="Predicted Class: ", font=font_result, bg="#f0f8ff", fg="#333")
label_result.pack(pady=10)

label_remedies = Label(root, text="", font=font_remedies, bg="#f0f8ff", fg="#444", justify="left", wraplength=400)
label_remedies.pack(pady=10)

footer = Label(root, text="Press F11 for Fullscreen | Esc to Exit Fullscreen | Powered by ML", font=("Helvetica", 10, "italic"), bg="#f0f8ff", fg="#555")
footer.pack(side="bottom", pady=10)

#Set window dimensions
root.geometry("600x700")
root.mainloop()




