# 🚨 Credit Card Fraud Detection with Random Forest

🎯 **Mục tiêu**: Dự đoán giao dịch thẻ tín dụng có phải là gian lận không, với Random Forest.  
Dữ liệu có tỷ lệ gian lận cực thấp (~0.17%), gây khó khăn cho việc huấn luyện mô hình.

---

🎥 **Đây là project được tôi thực hiện trong chuỗi livestream "HỌC ML CÙNG GÀ AI"**  
🕒 Tổng thời lượng: 7 buổi – 22 tiếng học sâu từng bước từ A đến Z.

📺 Xem toàn bộ playlist tại đây: 👉 [YouTube Playlist - Học ML cùng Gà AI](https://www.youtube.com/playlist?list=PLFOcj4yNRTxN2ZDHXDH16chYkiIuYlz46)

💻 Chạy code trực tiếp trên Google Colab: 👉 [Mở Notebook trên Colab](https://colab.research.google.com/drive/1HhiniuKlntMVeB8vy5SlnzgCK1Z_Pytr?usp=sharing)

---

## 📁 Dataset
- Nguồn: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 dòng, 492 gian lận (fraud)
- Gồm các đặc trưng ẩn V1 đến V28 (PCA), `Time`, `Amount`, và nhãn `Class`

## 🔍 Pipeline thực hiện (10 bước bài bản)
1. Khám phá dữ liệu (EDA)
2. Trực quan hóa dữ liệu
3. Tiền xử lý (`Amount`, `Time`)
4. Xử lý mất cân bằng (Undersampling, SMOTE, kết hợp)
5. Huấn luyện mô hình (Logistic, RF, XGBoost)
6. Trực quan kết quả (ROC, PR, confusion matrix)
7. Chọn mô hình tốt nhất theo PR-AUC
8. Tối ưu bằng GridSearchCV
9. Tối ưu threshold phân loại
10. Triển khai bằng Streamlit

## 🛠️ Kỹ thuật sử dụng
- 📊 Trực quan hóa: matplotlib, seaborn
- 🤖 Machine Learning: scikit-learn
- ⚖️ Mất cân bằng: imbalanced-learn (SMOTE, RandomUnderSampler)
- 🧪 Đánh giá: ROC-AUC, PR-AUC, F1, Recall, Confusion Matrix, Classification Report
- 🌐 Triển khai: Streamlit

## ▶️ Chạy thử project
```bash
# Clone repo
git clone https://github.com/Dinhk65/credit-card-fraud-detection-random-forest.git
cd credit-card-fraud-detection-random-forest

# Cài môi trường
pip install -r requirements.txt

# Chạy app Streamlit
streamlit run streamlit_app/app.py
