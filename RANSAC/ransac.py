import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def align_images():
    # Đọc ảnh đã chọn
    img1 = cv2.imread(entry_img1.get())  # Ảnh bị nghiêng
    img2 = cv2.imread(entry_img2.get())  # Ảnh gốc (chuẩn)

    # Trích xuất đặc trưng bằng ORB
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    # Ghép đặc trưng bằng Brute-Force Hamming
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = sorted(matcher.match(des1, des2), key=lambda x: x.distance)

    # Giữ lại 15% match tốt nhất
    good_matches = matches[:int(len(matches) * 0.15)]

    # Tạo 2 tập điểm tương ứng
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Tìm homography dùng RANSAC
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    # Warp ảnh 1 về tọa độ ảnh 2
    aligned = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    cv2.imwrite('aligned.jpg', aligned)

    # Hiển thị ảnh align
    show_image(aligned)

def browse_file(entry):
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)

def show_image(img):
    # Hiển thị ảnh trong giao diện
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.configure(image=img_tk)
    panel.image = img_tk

# === Giao diện Tkinter ===
root = tk.Tk()
root.title("Feature-Based Image Alignment (RANSAC)")
root.geometry("500x600")

tk.Label(root, text="Path ảnh nghiêng (template_warped):").pack()
entry_img1 = tk.Entry(root, width=50)
entry_img1.pack()
tk.Button(root, text="Chọn ảnh", command=lambda: browse_file(entry_img1)).pack()

tk.Label(root, text="Path ảnh gốc (template):").pack()
entry_img2 = tk.Entry(root, width=50)
entry_img2.pack()
tk.Button(root, text="Chọn ảnh", command=lambda: browse_file(entry_img2)).pack()

tk.Button(root, text="Align Ảnh (RANSAC)", command=align_images, bg="lightblue").pack(pady=10)
panel = tk.Label(root)
panel.pack()

root.mainloop()
