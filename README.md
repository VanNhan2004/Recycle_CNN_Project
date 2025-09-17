# ♻️ Ứng dụng Phân loại Rác thải Tái chế

![Banner](a04c7671-4217-4fd3-af27-f183b6bcb6a9.png)

## 🎯 Giới thiệu
Đây là đồ án tốt nghiệp của sinh viên **Nguyễn Văn Nhân – MSSV 2200002045 – Đại học Nguyễn Tất Thành**.  
Ứng dụng sử dụng **Mạng Nơ-ron Tích Chập (Convolutional Neural Network - CNN)** để **phân loại rác thải tái chế** thành 5 nhóm chính:  

- 📦 **Bìa cứng**  
- 🥂 **Thủy tinh**  
- 🥫 **Kim loại**  
- 📄 **Giấy**  
- 🍼 **Nhựa**

Ứng dụng được xây dựng bằng **TensorFlow/Keras** cho phần huấn luyện mô hình và **Streamlit** cho giao diện demo.

---

## 📂 Cấu trúc thư mục
```
recycle_cnn_project/
│── app.py                 # Ứng dụng Streamlit
│── requirements.txt       # Thư viện cần thiết
│── recycle_cnn_project/
│   │── data_processing.py
│   │── model.py
│   │── train.py
│   │── predict.py
│   │── evaluate.py
│   │── data/              # Dữ liệu (train/test/val)
│   │── models/            # Model đã train (.keras)
```

---

## 🧠 Kiến trúc mô hình CNN
Mô hình CNN được xây dựng với 3 lớp Convolution + Pooling và 2 lớp Dense:  

![Kiến trúc CNN](a5fcac8d-57ec-446f-b3d9-b881069d21a1.png)

- **Conv2D + MaxPooling2D** (trích xuất đặc trưng)  
- **Flatten** (chuyển sang vector 1 chiều)  
- **Dense + Dropout** (học đặc trưng và tránh overfitting)  
- **Dense(5, softmax)** (dự đoán 5 lớp đầu ra)  

---

## 🚀 Cài đặt & Chạy chương trình

### 1. Clone repo
```bash
git clone https://github.com/username/recycle_cnn_project.git
cd recycle_cnn_project
```

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 3. Huấn luyện mô hình
```bash
python recycle_cnn_project/train.py
```
👉 Model tốt nhất sẽ được lưu tại `models/best_model.keras`

### 4. Chạy ứng dụng Streamlit
```bash
streamlit run app.py
```

---

## 🖥️ Giao diện ứng dụng
### Trang chủ
![Trang chủ](5e491509-28e1-4d43-819c-4d39ffe7f69d.png)

### Upload & Dự đoán
![Upload & Predict](a04c7671-4217-4fd3-af27-f183b6bcb6a9.png)

### Thống kê kết quả
![Evaluation](a5fcac8d-57ec-446f-b3d9-b881069d21a1.png)

---

## 📊 Kết quả mô hình
Kết quả trên tập test:  

- 🎯 Accuracy: ~76%  
- 📊 Classification Report & Confusion Matrix hiển thị trực tiếp trên giao diện Streamlit  

---

## 👨‍🎓 Thông tin sinh viên
- **Họ và tên**: Nguyễn Văn Nhân  
- **MSSV**: 2200002045  
- **Trường**: Đại học Nguyễn Tất Thành   

---
✍️ *Repo này được xây dựng nhằm trình bày sản phẩm cuối cùng của đồ án tốt nghiệp.*
