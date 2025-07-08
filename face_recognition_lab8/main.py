import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib

pca, face_vectors_db, labels_db = joblib.load("eigenface_data.pkl")
face_size = (100, 100)
threshold = 0.1
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def process_image(image_path):
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, face_size).flatten().reshape(1, -1)

        # Vector PCA of new face
        vector = pca.transform(face_resized)[0]
        # Cosine similarity
        norms_db = np.linalg.norm(face_vectors_db, axis=1)
        norm_q = np.linalg.norm(vector)
        sims = np.dot(face_vectors_db, vector) / (norms_db * norm_q + 1e-8)

        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score > threshold:
            name = labels_db[best_idx]
            label = f"{name} ({best_score:.2f})"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame

def choose_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        processed_img = process_image(file_path)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(processed_img)
        im_pil.thumbnail((600, 600))  
        imgtk = ImageTk.PhotoImage(image=im_pil)

        output_label.configure(image=imgtk)
        output_label.image = imgtk

root = tk.Tk()
root.title("Face Recognition")
root.geometry("700x700")
btn = tk.Button(root, text="Choose Image", command=choose_image, font=("Arial", 14))
btn.pack(pady=20)
output_label = tk.Label(root)
output_label.pack()
root.mainloop()
