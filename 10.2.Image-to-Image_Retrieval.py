# II.2. Pixel-wise Retrieval
# II.2.1. So sánh độ tương đồng dựa trên pixel

# hai hình ảnh được đưa về cùng kích thước, sau đó độ tương đồng được tính toán bằng
# cách so sánh các giá trị pixel tương ứng trên từng kênh màu. Các phép đo khoảng cách phổ biến
# như L1 distance hoặc L2 distance có thể được sử dụng để lượng hóa mức độ khác biệt giữa hai
# hình ảnh.

import random
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

ROOT = "D:/python folder/animal"

# Xây dựng bộ dữ liệu
# đọc các tấm ảnh và tiền xử lý bằng cách đưa các tấm ảnh về cùng kích thước là 128×128.
def load_image(path,size = (128,128)):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load {path}")
    
    img = cv2.resize(img,size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)

def build_database(root_dir, size=(128,128)):
    images = []
    labels = []
    paths = []
    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in sorted(os.listdir(cls_path)):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            path = os.path.join(cls_path,fname)
            img = load_image(path, size)
            images.append(img)
            labels.append(cls)
            paths.append(path)
    return images, labels, paths

db_images, db_labels, db_paths = build_database(ROOT)

# L1 và L2 distances
def l1_distance(a,b):
    # """Pixel-wise L1 distance for RGB images"""
    return np.sum(np.abs(a-b))

def l2_distance(a,b):
    # """Pixel-wise L2 distance for RGB images"""
    diff = a - b
    return np.sqrt(np.sum(diff * diff))

# Hàm truy vấn
def retrieve(query_img, db_images, db_labels, db_paths, distance_fn, top_k = 5):
    results = []
    for img, label, path in zip(db_images, db_labels, db_paths):
        dist = distance_fn(query_img, img)
        results.append((path, label, dist))
    results.sort(key = lambda x : x[2])
    return results[:top_k]

# Hàm trực quan kết quả
def visualize_retrieval(query_img, results, title, top_k = 5):
    plt.figure(figsize=(3* (top_k + 1), 4))

    # Query image
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(query_img.astype("uint8"))
    plt.title("Query")
    plt.axis("off")

    # Retrieved images
    for i, (path, label, dist) in enumerate(results[:top_k], start = 2):
        img = load_image(path).astype("uint8")

        plt.subplot(1,top_k + 1, i)
        plt.imshow(img)
        plt.title(f"{label}\n{dist:.1f}")
        plt.axis("off")
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# Thực hiện truy vấn
# img_path = os.path.join(ROOT, 'bear', '3.jpg')
img_path = "D:/python folder/animal/bear/3.jpg"
query_img = load_image(img_path)
# L1
top_l1 = retrieve(query_img,  db_images, db_labels, db_paths, l1_distance, top_k=5)
visualize_retrieval(query_img, top_l1, title="Top-5 Retrieval (L1 Distance)")

# L2
top_l2 = retrieve(query_img, db_images, db_labels, db_paths, l2_distance, top_k=5)
visualize_retrieval(query_img, top_l2, title="Top-5 Retrieval (L2 Distance)")




### II.3. Feature extraction (trích xuất đặc trưng)
# Hàm trích xuất đặc trưng ảnh
def extract_color_histogram(img, bins=(8,8,8)):
    hist = cv2.calcHist(
        [img.astype(np.uint8)],
        channels=[0,1,2],
        mask=None,
        histSize=bins,
        ranges=[0, 256, 0, 256, 0, 256]
    )
    # Normalize histogram
    hist = cv2.normalize(hist, hist).flatten()
    # normalize(): thực hiện việc chuẩn hóa.(bạn không thể so sánh hai bức ảnh khác kích thước.)
    # Sau khi chuẩn hóa, histogram đại diện cho tỉ lệ xuất hiện của màu sắc chứ không phải số lượng pixel tuyệt đối.

    # .flatten(): Chuyển mảng histogram từ dạng đa chiều (8 x 8 x 8) thành một mảng 1 chiều (gồm 512 phần tử liên tiếp).
    # để dễ dàng đưa vào các hàm tính khoảng cách (L1, L2) đã viết ở phần trước.

    return hist

# Ta thực hiện trích xuất đặc trưng cho toàn bộ dữ liệu đầu vào trước, thay vì phải thực hiện lại
# mỗi khi query

# Xây dựng dữ liệu đã qua trích xuất đặc trưng
def build_hist_database(db_images, db_labels, db_paths, bins=(8,8,8)):
    features = []

    for img in db_images:
        hist = extract_color_histogram(img, bins)
        features.append(hist)

    return np.array(features), db_labels, db_paths

db_hists, db_labels, db_paths = build_hist_database(db_images, db_labels, db_paths)

# Tiếp theo, ta chỉ cần xây dựng lại hàm retrieval và tái sử dụng các hàm khác như L1_distance,
# L2_distance hay visualize_retrieval.

# Hàm truy vấn với Color Histogram
def retrieve_hist(
        query_img,
        db_features,
        db_labels,
        db_paths,
        distance_fn,
        bins=(8,8,8),
        top_k=5
):
    query_hist = extract_color_histogram(query_img, bins)
    results = []

    for feat, label, path in zip(db_features, db_labels, db_paths):
        dist = distance_fn(query_hist, feat)
        results.append((path, label, dist))

    results.sort(key=lambda x : x[2])
    return results[:top_k]

# Truy vấn với Color Histogram
# Query image
img_path = "D:/python folder/animal/bear/3.jpg"
query_img = load_image(img_path)

# Retrieve using color histograms
top_l1 = retrieve_hist(query_img, db_hists, db_labels, db_paths, distance_fn = l1_distance, top_k = 5)
top_l2 = retrieve_hist(query_img, db_hists, db_labels, db_paths, distance_fn=l2_distance, top_k=5)

# Visualize
visualize_retrieval(query_img, top_l1, "Color Histogram Retrieval (L1)")
visualize_retrieval(query_img, top_l2, "Color Histogram Retrieval (L2)")

# II.5. Gợi ý mở rộng project
# Project có thể được mở rộng theo nhiều hướng khác nhau. Trước hết, ta có thể áp dụng các
# phương pháp trích xuất đặc trưng ảnh đa dạng hơn, chẳng hạn như Color Moments, Local
# Binary Patterns (LBP), Histogram of Oriented Gradients (HOG) hoặc các đặc trưng
# dựa trên hình dạng. Ngoài ra, việc kết hợp nhiều loại đặc trưng (feature fusion) cũng là một
# hướng tiếp cận hiệu quả, giúp khai thác đồng thời thông tin về màu sắc, kết cấu và hình dáng
# của ảnh.
# Đối với bài toán Image-to-Image Retrieval, các phương pháp content-based nhìn chung
# mang lại hiệu quả vượt trội so với các phương pháp dựa thuần túy trên pixel. Do đó, một hướng
# mở rộng quan trọng là áp dụng các mô hình học sâu để trích xuất đặc trưng ở mức ngữ nghĩa
# cao hơn. Các mô hình như ResNet, Vision Transformer (ViT) hoặc đặc biệt là CLIP có
# khả năng ánh xạ ảnh vào không gian đặc trưng giàu ngữ nghĩa, từ đó cải thiện đáng kể chất
# lượng truy vấn và độ chính xác của hệ thống retrieval.