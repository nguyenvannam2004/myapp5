# Import các thư viện cần thiết
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import joblib
from imblearn.over_sampling import SMOTE
import numpy as np

# Bước 1: Đọc dữ liệu bệnh tim từ file CSV 
data = pd.read_csv('./mynewdata.csv')

# Bước 2: Xem trước dữ liệu
#print(data.head())

# Giả sử các đặc trưng (feature) của bạn là các cột từ 0 đến -1 và cột đích là 'target'
X = data.drop(columns=['target', 'STT'])  # Các đặc trưng đầu vào
y = data['target']  # Nhãn mục tiêu

# Chia dữ liệu: 80% cho huấn luyện, 20% cho xác thực và kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Chia 20% còn lại thành 50% cho xác thực và 50% cho kiểm tra
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# In số mẫu của các tập
print(f'Số mẫu của tập huấn luyện: {len(X_train)}')
print(f'Số mẫu của tập xác thực: {len(X_val)}')
print(f'Số mẫu của tập kiểm tra: {len(X_test)}')

# Cân bằng lớp với SMOTE trên tập huấn luyện
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Bước 5: Khởi tạo và huấn luyện mô hình Logistic Regression
model = LogisticRegression(max_iter=1500)
model.fit(X_train_resampled, y_train_resampled)

# Dự đoán và đánh giá trên tập xác thực
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Dộ chính xác trên tập xác thực: {val_accuracy:.10f}')

# Dự đoán và đánh giá trên tập kiểm tra
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Dộ chính xác trên tập kiểm tra: {test_accuracy:.10f}')

# Lưu mô hình
joblib.dump(model, "logistic_model.pkl")

# Tính và vẽ ROC Curve (chỉ áp dụng với bài toán nhị phân)
if len(set(y)) == 2:
    # Dự đoán xác suất
    y_test_proba = model.predict_proba(X_test)[:, 1]  # Xác suất của lớp 1

    # Tính toán ROC và AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    auc = roc_auc_score(y_test, y_test_proba)
    print(f'Giá trị AUC: {auc:.2f}')
    
    # Vẽ ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Đường tham chiếu
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Vẽ biểu đồ phân phối xác suất dự đoán
plt.figure(figsize=(8, 6))
sns.histplot(y_test_proba, bins=20, kde=True, color='blue')
plt.title("Distribution of Prediction Probabilities")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.show()

# Tính toán và in ra ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Ma trận nhầm lẫn:")
print(conf_matrix)

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Không có bệnh', 'Có bệnh'], yticklabels=['Không có bệnh', 'Có bệnh'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')
plt.show()

class_report = classification_report(y_test, y_test_pred)
print('Báo cáo phân loại:')
print(class_report)

# Dữ liệu mới bạn muốn dự đoán
new_data = np.array([57, 0, 1, 130, 236, 0, 0, 174, 0, 0, 1, 1, 2])

# Định dạng lại dữ liệu thành mảng 2D
new_data_reshaped = new_data.reshape(1, -1)

y_val_pred = model.predict(new_data_reshaped)
print(y_val_pred)



threshold = 0.4

# Bước 7: Dự đoán với ngưỡng mới
y_test_pred_new_threshold = (y_test_proba >= threshold).astype(int)

# Bước 8: Tính toán độ chính xác và các chỉ số khác với ngưỡng mới
new_test_accuracy = accuracy_score(y_test, y_test_pred_new_threshold)
print(f'Dộ chính xác trên tập kiểm tra (với ngưỡng {threshold}): {new_test_accuracy:.10f}')

# Ma trận nhầm lẫn với ngưỡng mới
new_conf_matrix = confusion_matrix(y_test, y_test_pred_new_threshold)
print("Ma trận nhầm lẫn với ngưỡng mới:")
print(new_conf_matrix)

# Vẽ lại ma trận nhầm lẫn với ngưỡng mới
plt.figure(figsize=(8, 6))
sns.heatmap(new_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Không có bệnh', 'Có bệnh'], yticklabels=['Không có bệnh', 'Có bệnh'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title(f'Ma trận nhầm lẫn với ngưỡng {threshold}')
plt.show()

# In báo cáo phân loại với ngưỡng mới
new_class_report = classification_report(y_test, y_test_pred_new_threshold)
print('Báo cáo phân loại với ngưỡng mới:')
print(new_class_report)

# Dữ liệu mới bạn muốn dự đoán
new_data = np.array([57, 0, 1, 130, 236, 0, 0, 174, 0, 0, 1, 1, 2])

# Định dạng lại dữ liệu thành mảng 2D
new_data_reshaped = new_data.reshape(1, -1)

# Dự đoán với dữ liệu mới
y_val_pred = model.predict(new_data_reshaped)
print(f'Dự đoán cho dữ liệu mới: {y_val_pred}')


