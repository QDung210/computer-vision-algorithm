import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

# ================================================================================================================
# ===================================== Thuật toán snake active contours =========================================
# ================================================================================================================
def preprocess(path='image.jpg', size=(500, 500), invert=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    if invert:
        img = 255 - img 
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
    return img, grad.astype(np.float32)
def create_circle_contour(center=(260, 260), radius=240, points=500, shape=(500, 500)):
    angles = np.linspace(0, 2*np.pi, points, endpoint=False)
    return [
        (int(np.clip(center[0] + radius*np.cos(a), 0, shape[1]-1)),
         int(np.clip(center[1] + radius*np.sin(a), 0, shape[0]-1)))
        for a in angles
    ]
def get_combined_score(grad, img, x, y, alpha=0.7, beta=0.3):
    g = map_coordinates(grad, [[y], [x]], order=1, mode='nearest')[0]
    i = map_coordinates(img.astype(np.float32), [[y], [x]], order=1, mode='nearest')[0]
    return alpha * g + beta * i

def find_edges(contour, grad, img, center=(260, 260), steps=100):
    dest = []
    for x0, y0 in contour:
        best_score = -1
        best_point = (x0, y0)
        for t in np.linspace(0, 1, steps):
            x = (1 - t) * x0 + t * center[0]
            y = (1 - t) * y0 + t * center[1]
            if 0 <= x < grad.shape[1] and 0 <= y < grad.shape[0]:
                score = get_combined_score(grad, img, x, y, alpha=0.6, beta=0.4)
                if score > best_score:
                    best_score = score
                    best_point = (int(x), int(y))
        dest.append(best_point)
    return dest

# ================================================================================================================
# ============================================= Thuật toán watershed =============================================
# ================================================================================================================
def watershed():
    img = cv2.imread("leaf.jpg")
    print(img.shape)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title("Original Grayscale")
    plt.subplot(232)
    _, imgThreshold = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    plt.imshow(imgThreshold, cmap='gray')
    plt.title("Thresholded Image")
    plt.subplot(233)
    kernel = np.ones((3,3), np.uint8)
    imgDilate = cv2.morphologyEx(imgThreshold, cv2.MORPH_DILATE, kernel)
    plt.imshow(imgDilate)
    plt.title("Dilated Image")
    plt.subplot(234)
    distTrans = cv2.distanceTransform(imgDilate, cv2.DIST_L2,5)
    plt.imshow(distTrans)
    plt.title("Distance Transform")
    plt.subplot(235)
    _, distThresh = cv2.threshold(distTrans, 15, 255, cv2.THRESH_BINARY)
    plt.imshow(distThresh)
    plt.title("Distance Threshold")
    plt.subplot(236)
    distThresh = np.uint8(distThresh)
    _, labels = cv2.connectedComponents(distThresh)
    plt.imshow(labels)
    plt.title("Connected Components")
    plt.figure()
    plt.subplot(121)
    labels = np.int32(labels)
    labels = cv2.watershed(imgRGB, labels)
    plt.imshow(labels)
    plt.title("Watershed Labels")
    plt.subplot(122)
    # Tạo mặt nạ ranh giới và giãn nở để làm dày đường đỏ
    boundary_mask = (labels == -1).astype(np.uint8) 
    kernel = np.ones((5, 5), np.uint8)  
    boundary_mask = cv2.dilate(boundary_mask, kernel, iterations=1)
    imgRGB[boundary_mask != 0] = [255, 0, 0]
    plt.imshow(imgRGB)
    plt.title("Final Result with Boundaries")
    plt.show()


# ================================================================================================================
# ============================================= Thuật toán K-Mean ================================================
# ================================================================================================================
def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

def kmeans(image_path="leaf.jpg", n_clusters=4, max_iter=100, tol=1e-4):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    np.random.seed(42)
    centers = pixels[np.random.choice(pixels.shape[0], n_clusters, replace=False)]

    for iteration in range(max_iter):
        distances = np.array([euclidean_distance(pixels, c) for c in centers]) 
        labels = np.argmin(distances, axis=0) 
        new_centers = np.array([pixels[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
                                for i in range(n_clusters)])
        shift = np.linalg.norm(centers - new_centers, axis=1).max()
        if shift < tol:
            break
        centers = new_centers
    segmented_pixels = centers[labels].reshape(img_rgb.shape).astype(np.uint8)
    segmented_bgr = cv2.cvtColor(segmented_pixels, cv2.COLOR_RGB2BGR)
    return segmented_bgr

# ================================================================================================================
# ============================================= Thuật toán Mean Shift ============================================
# ================================================================================================================
def mean_shift_segmentation(image_path="leaf.jpg"):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    flat_image = lab_image.reshape((-1, 3))
    height, width, _ = img.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    flat_image_with_coordinates = np.column_stack([
        flat_image,
        x.flatten(),
        y.flatten()
    ])
    bandwidth = estimate_bandwidth(flat_image_with_coordinates, quantile=0.2, n_samples=500)
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    mean_shift.fit(flat_image_with_coordinates)
    labels = mean_shift.labels_
    unique_labels = np.unique(labels)
    original_flat_rgb = img.reshape(-1, 3)
    segmented_rgb_flat = np.zeros_like(original_flat_rgb, dtype=np.uint8)
    for label in unique_labels:
        mask = (labels == label)
        mean_color = original_flat_rgb[mask].mean(axis=0)
        segmented_rgb_flat[mask] = mean_color

    segmented_image_rgb = segmented_rgb_flat.reshape((height, width, 3))
    plt.figure(figsize=(10, 5))
    plt.imshow(segmented_image_rgb)
    plt.title(f"Mean Shift")
    plt.axis('off')
    plt.show()
# ================================================================================================================
# ============================================= Hàm main để vẽ hình ==============================================
# ================================================================================================================
if __name__ == "__main__":

    # ===================================== Thuật toán snake active contours =========================================
    img, grad = preprocess()
    contour = create_circle_contour()
    edges = find_edges(contour, grad, img)
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    for t in np.linspace(0, 1, 50):
        frame = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        for (x0, y0), (x1, y1) in zip(contour, edges):
            x, y = int((1 - t) * x0 + t * x1), int((1 - t) * y0 + t * y1)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        ax.clear()
        ax.imshow(frame)
        ax.set_title("Snake Algorithm") 
        ax.axis("off")
        plt.pause(0.05)
    plt.ioff()
    plt.show()

    # ============================================= Thuật toán watershed =============================================
    watershed()

    # =============================================== Thuật toán K-mean ==============================================
    result = kmeans("leaf.jpg", n_clusters=4)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    # =============================================== Thuật toán Mean Shift ==============================================
    mean_shift_segmentation()

