import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import joblib

face_size = (100, 100)      
images_path = "images"  

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Prepare storage for images and labels
images = []
labels = []
for person_name in os.listdir(images_path):
    person_path = os.path.join(images_path, person_name)
    if os.path.isdir(person_path):
        for image_file in os.listdir(person_path):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_path, image_file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Detect face
                faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in faces:
                    face = img[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, face_size)
                    images.append(face_resized.flatten())  # Flatten to 1D vector
                    labels.append(person_name)
                    print(f"Added: {image_path}")
                    break  # Only process the first detected face

# Convert to NumPy array
images_np = np.array(images)
# Train PCA on all face vectors
n_components = min(500, len(images))  
pca = PCA(n_components=n_components, whiten=True)
pca.fit(images_np)
# Project all face vectors into PCA space
face_vectors = pca.transform(images_np)
# Save the PCA model, vectors, and labels
joblib.dump((pca, face_vectors, labels), "eigenface_data.pkl")
print("Training complete. Data saved to 'eigenface_data.pkl'")
