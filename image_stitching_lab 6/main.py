import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def stitch_images():
    # Đọc ảnh đầu vào
    img1 = cv2.imread(entry_img1.get()) 
    img2 = cv2.imread(entry_img2.get())

    # Chuyển sang grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Phát hiện keypoints và descriptors bằng ORB
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    good_matches = matcher.match(des1, des2)
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Tạo các cặp điểm keypoint tương ứng
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Tính homography dùng RANSAC
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    # Kích thước ảnh đầu vào
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Warp ảnh 1 vào không gian ảnh 2
    warped_img1 = cv2.warpPerspective(img1, H, (w1 + w2, h2))
    warped_img1[0:h2, 0:w2] = img2

    # Lưu và hiển thị ảnh kết quả
    cv2.imwrite("stitched_result.jpg", warped_img1)
    show_image(warped_img1)

def browse_file(entry):
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)

def show_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.configure(image=img_tk)
    panel.image = img_tk

root = tk.Tk()
root.title("Image Stitching with ORB + RANSAC")
root.geometry("500x600")

tk.Label(root, text="Path ảnh bên trái:").pack()
entry_img1 = tk.Entry(root, width=50)
entry_img1.pack()
tk.Button(root, text="Chọn ảnh", command=lambda: browse_file(entry_img1)).pack()

tk.Label(root, text="Path ảnh bên phải:").pack()
entry_img2 = tk.Entry(root, width=50)
entry_img2.pack()
tk.Button(root, text="Chọn ảnh", command=lambda: browse_file(entry_img2)).pack()

tk.Button(root, text="Stitch ảnh (RANSAC)", command=stitch_images, bg="lightgreen").pack(pady=10)
panel = tk.Label(root)
panel.pack()

root.mainloop()
