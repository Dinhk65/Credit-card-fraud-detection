import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

loaded_model = joblib.load(r"C:\Users\User\Downloads\credit_card_fraud_detection_rf.pkl")
print("Mô hình đã được tải thành công!")

st.set_page_config(page_title="Fraud Detection", layout="wide")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("CREDIT CART FRAUD DETECTION")

# Load model
# Bước 1: Tải file CSV
st.sidebar.header("1️⃣ Tải file CSV dữ liệu mới")
data_file = st.sidebar.file_uploader("Tải dữ liệu: ", type=["csv","xlsx"])

# Lựa chọn ngưỡng
col1, col2 = st.sidebar.columns(2)
with col1:
    threshold_slider = st.slider("Ngưỡng 1 (kéo thả)", 0.0, 1.0, 0.5, 0.01)
with col2:
    threshold = st.number_input("Hoặc nhập giá trị", 0.0, 1.0, threshold_slider, 0.01)

# Đọc dữ liệu
if data_file is not None:
    new_data = pd.read_csv(data_file, encoding= 'utf-8')
    st.write("📊 DỮ LIỆU TẢI LÊN:")
    st.dataframe(new_data.head())

    st.success("✅ TẢI DỮ LIỆU THÀNH CÔNG!")

    # Hiển thị sơ lược
    with st.expander("🔍 THÔNG TIN DỮ LIỆU"):
        st.write(new_data.head())

    with st.expander("📊THỐNG KÊ MÔ TẢ: df.describe()"):
        st.write(new_data.describe())

    # Xử lý dữ liệu
    with st.expander("🔍XỬ LÝ DỮ LIỆU "):
        st.write("Bắt đầu Xử lý dữ liệu:")

        df = new_data.copy()

        # Xóa các dòng chứa NaN
        df = df.dropna()
        st.write("Đã Xóa Mising Value!")

        # Tạo đặc trưng mới
        df['Hour'] = (df['Time'] // 3600) % 24
        st.write("Đã thêm đặc trưng Hour vào Data!")

        df['Is_Night'] = df['Hour'].apply(lambda x: 1 if x < 6 else 0)
        st.write("Đã thêm đặc trưng Is_Night vào Data!")

        df['High_Amount'] = df['Amount'].apply(lambda x: 1 if x > 1000 else 0)
        st.write("Đã thêm đặc trưng High_Amount vào Data!")

        bins = [0, 100, 500, 1000, 5000, 2000000]
        labels = [0, 1, 2, 3, 4]
        df['Amount_Bins'] = pd.cut(df['Amount'], bins=bins, labels=labels, include_lowest=True)
        df['Amount_Bins'] = df['Amount_Bins'].astype(int)
        st.write("Đã thêm đặc trưng Amount_Bins vào Data!")

        df['High_Amount_at_Night'] = df['Is_Night'] * df['High_Amount']
        st.write("Đã thêm đặc trưng High_Amount_at_Night vào Data!")

    st.success("✅ ĐÃ THÊM ĐẶC TRƯNG VÀO DATA!")

    # Lấy danh sách features mà mô hình đã được huấn luyện
    required_features = loaded_model.feature_names_in_

    # Chọn chỉ các cột cần thiết từ dữ liệu mới
    X_predict = df[required_features]

    # Dự đoán
    # Xử lý dữ liệu
    with st.expander("🔍DỰ ĐOÁN"):
        y_proba = loaded_model.predict_proba(X_predict)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        st.write("Đã tính y_pred và y_prob!")

        df['fraud_probability'] = y_proba
        df['is_fraud'] = y_pred

        st.write("📈 Kết quả dự đoán:")
        st.dataframe(df[['fraud_probability', 'is_fraud']].head())

    with st.expander("🔍TRỰC QUAN HÓA KẾT QUẢ:"):
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            sns.histplot(y_proba, bins=30, kde=True, ax=ax, color='skyblue')
            ax.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold}")
            ax.set_title("Phân bố xác suất dự đoán")
            ax.legend()
            st.pyplot(fig)

        with col2:
            fraud_counts = df['is_fraud'].value_counts().rename({0: 'Không gian lận', 1: 'Gian lận'})
            st.write("Tỷ lệ phân loại:")
            st.write(fraud_counts)
            fig2, ax2 = plt.subplots()
            ax2.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff6666'])
            ax2.axis('equal')
            st.pyplot(fig2)

        # Cho tải kết quả
        csv_output = df.to_csv(index=False)
        st.download_button("📥 Tải kết quả dự đoán", data=csv_output, file_name="du_doan_fraud.csv", mime="text/csv")
