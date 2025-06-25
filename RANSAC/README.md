# Image Alignment Project

Dự án này cung cấp hai công cụ chính để xử lý và căn chỉnh hình ảnh sử dụng OpenCV và Python.

## 📋 Mô tả

Dự án bao gồm hai script chính:
- **rotate.py**: Tạo ảnh bị biến dạng từ ảnh gốc để mô phỏng việc chụp ảnh bị nghiêng
- **ransac.py**: Sử dụng thuật toán RANSAC để căn chỉnh lại ảnh bị nghiêng về vị trí ban đầu


## 📦 Yêu cầu hệ thống

```bash
pip install -r requirements.txt
```

## 🔧 Cách sử dụng

### Bước 1: Tạo ảnh mẫu bị biến dạng

```bash
python rotate.py
```

**Yêu cầu**: Đặt file `template.jpg` trong cùng thư mục với script.

**Kết quả**: Tạo ra file `template_warped.jpg` - ảnh bị biến dạng.

### Bước 2: Căn chỉnh ảnh bằng RANSAC

```bash
python ransac.py
```

**Các bước thực hiện**:
1. Chọn ảnh bị nghiêng (template_warped.jpg)
2. Chọn ảnh gốc (template.jpg)
3. Nhấn "Align Ảnh (RANSAC)"
4. Xem kết quả căn chỉnh trên giao diện

**Kết quả**: Tạo ra file `aligned.jpg` - ảnh đã được căn chỉnh.

## 📁 Cấu trúc thư mục

```
project/
├── rotate.py           # Script tạo ảnh biến dạng
├── ransac.py          # Script căn chỉnh ảnh với giao diện
├── template.jpg       # Ảnh gốc (cần chuẩn bị)
├── template_warped.jpg # Ảnh bị biến dạng (được tạo)
├── aligned.jpg        # Ảnh đã căn chỉnh (được tạo)
└── README.md         # File hướng dẫn này
```

## ⚙️ Chi tiết kỹ thuật

### Thuật toán sử dụng:

1. **ORB (Oriented FAST and Rotated BRIEF)**:
   - Trích xuất đặc trưng từ ảnh
   - Tạo ra 500 keypoints cho mỗi ảnh

2. **Brute-Force Hamming Matcher**:
   - Ghép đặc trưng giữa hai ảnh
   - Sắp xếp theo khoảng cách tăng dần

3. **RANSAC (Random Sample Consensus)**:
   - Tìm ma trận homography tối ưu
   - Loại bỏ các điểm nhiễu (outliers)
