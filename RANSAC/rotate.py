import cv2
import numpy as np

img = cv2.imread('template.jpg')
if img is None:
    print("❌ Không đọc được ảnh.")
    exit()

h, w = img.shape[:2]

pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
pts2 = np.float32([
    [200, 240],            # Top-left lệch mạnh hơn
    [w - 200, 0],          # Top-right bị kéo vào nhiều hơn
    [160, h - 240],        # Bottom-left
    [w - 300, h - 100]     # Bottom-right
])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
warped = cv2.warpPerspective(img, matrix, (w, h))

cv2.imwrite('template_warped.jpg', warped)
