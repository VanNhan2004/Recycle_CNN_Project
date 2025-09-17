import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import time

# ==============================
# Config UI
# ==============================
st.set_page_config(
    page_title="Phân loại rác thải tái chế",
    page_icon="🗑️",
    layout="wide"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .sidebar .sidebar-content { background-color: #e9eff5; }
    .stButton>button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Load model
# ==============================
@st.cache_resource
def load_trash_model():
    return load_model("models/best_model.keras")

model = load_trash_model()
labels = ["Bìa cứng","Thủy tinh","Kim loại","Giấy","Nhựa"]

# ==============================
# Sidebar menu
# ==============================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1041/1041373.png", width=80)
st.sidebar.title("🗂️ Điều hướng")

menu = ["🏠 Trang chủ", "📤 Upload & Dự đoán", "📊 Thống kê", "ℹ️ Giới thiệu"]
choice = st.sidebar.radio("Chọn chức năng:", menu)

# ==============================
# Trang chủ
# ==============================
if choice == "🏠 Trang chủ":
    st.title("♻️ Ứng dụng Phân loại Rác thải Tái chế")
    st.markdown("""
    ### 👨‍💻 Đồ án tốt nghiệp  
    Ứng dụng sử dụng **Mạng Nơ-ron Tích Chập (CNN)** để phân loại rác thải tái chế thành các nhóm:  
    - 📦 Bìa cứng  
    - 🥂 Thủy tinh  
    - 🥫 Kim loại  
    - 📄 Giấy  
    - 🍼 Nhựa  
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/679/679922.png", width=250)

# ==============================
# Upload & Dự đoán
# ==============================
elif choice == "📤 Upload & Dự đoán":
    st.title("📤 Upload ảnh và Dự đoán 🔮")

    uploaded_file = st.file_uploader("Chọn ảnh từ máy tính:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Ảnh đã tải lên", use_column_width=True)

        if st.button("🚀 Dự đoán ngay"):
            with st.spinner("Đang xử lý ảnh..."):
                time.sleep(1.5)

                # Tiền xử lý ảnh
                img_resized = img.resize((224, 224))
                x = np.expand_dims(image.img_to_array(img_resized), axis=0) / 255.0

                # Dự đoán
                preds = model.predict(x)[0]
                pred_idx = np.argmax(preds)
                pred_label = labels[pred_idx]
                confidence = preds[pred_idx] * 100

            st.success(f"👉 Kết quả: **{pred_label}** ({confidence:.2f}%)")

            # Progress bar
            st.progress(int(confidence))

            # Biểu đồ xác suất
            st.subheader("📊 Xác suất từng lớp")
            st.bar_chart(pd.DataFrame(preds, index=labels, columns=["Xác suất"]))

# ==============================
# Thống kê
# ==============================
elif choice == "📊 Thống kê":
    st.title("📊 Thống kê hiệu suất mô hình")

    test_dir = "data/data_split/test"
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    y_pred = model.predict(test_generator, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    # Classification report
    report = classification_report(y_true, y_pred_classes, target_names=labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.background_gradient(cmap="Blues"))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

# ==============================
# Giới thiệu
# ==============================
elif choice == "ℹ️ Giới thiệu":
    st.title("ℹ️ Thông tin đồ án")
    st.markdown("""
    **Tên sinh viên:** Nguyễn Văn Nhân  
    **MSSV:** 2200002045  
    **Trường:** Đại học Nguyễn Tất Thành  
    **Đề tài:** Nhận diện & Phân loại rác thải tái chế bằng CNN  

    Ứng dụng này được xây dựng bằng **Streamlit** để minh họa sản phẩm cuối cùng của đồ án 🎓.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=150)
