import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Bước 1: Đọc dữ liệu bệnh tim từ file CSV 
data = pd.read_csv('./mynewdata.csv')

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

# Khởi tạo mô hình
rf_model = RandomForestClassifier(n_estimators=115, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=115, random_state=42)

# Sử dụng mô hình Voting Classifier để kết hợp các mô hình
ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('gb', gb_model)], voting='soft')

# Huấn luyện mô hình với dữ liệu đã cân bằng
ensemble_model.fit(X_train_resampled, y_train_resampled)

# Dự đoán trên tập xác thực
y_val_pred = ensemble_model.predict(X_val)

# Đánh giá mô hình trên tập xác thực
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Dộ chính xác trên tập xác thực: {val_accuracy:.10f}')

# Dự đoán trên tập kiểm tra
y_test_pred = ensemble_model.predict(X_test)

# Đánh giá mô hình trên tập kiểm tra
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Dộ chính xác trên tập kiểm tra: {test_accuracy:.10f}')

if len(set(y)) == 2:
    # Dự đoán xác suất
    y_test_proba = ensemble_model.predict_proba(X_test)[:, 1]  # Xác suất của lớp 1

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

# Ma trận nhầm lẫn cho tập kiểm tra
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Ma trận nhầm lẫn:")
print(conf_matrix)

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Không có bệnh', 'Có bệnh'], yticklabels=['Không có bệnh', 'Có bệnh'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn - Tập kiểm tra')
plt.show()

# Báo cáo phân loại cho tập kiểm tra
class_report = classification_report(y_test, y_test_pred)
print('Báo cáo phân loại cho tập kiểm tra:')
print(class_report)

# Dữ liệu mới bạn muốn dự đoán
new_data = np.array([51,1,0,140,299,0,1,173,1,1.6,2,0,3])

# Định dạng lại dữ liệu thành mảng 2D
new_data_reshaped = new_data.reshape(1, -1)

# Sử dụng dữ liệu đã định dạng lại để dự đoán
y_val_pred = ensemble_model.predict(new_data_reshaped)

print(y_val_pred)
joblib.dump(ensemble_model, "ensemble_model.pkl")



# Dự đoán xác suất
y_test_proba = ensemble_model.predict_proba(X_test)[:, 1]  # Xác suất của lớp 1

# Sử dụng ngưỡng 0.6 thay vì 0.5
y_test_pred_with_threshold = (y_test_proba >= 0.654).astype(int)

# Đánh giá mô hình trên tập kiểm tra với ngưỡng mới
test_accuracy_with_threshold = accuracy_score(y_test, y_test_pred_with_threshold)
print(f'Dộ chính xác trên tập kiểm tra với ngưỡng 0.6: {test_accuracy_with_threshold:.10f}')

# Ma trận nhầm lẫn cho tập kiểm tra với ngưỡng mới
conf_matrix_with_threshold = confusion_matrix(y_test, y_test_pred_with_threshold)
print("Ma trận nhầm lẫn với ngưỡng 0.6:")
print(conf_matrix_with_threshold)

# Vẽ ma trận nhầm lẫn với ngưỡng mới
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_with_threshold, annot=True, fmt='d', cmap='Blues', xticklabels=['Không có bệnh', 'Có bệnh'], yticklabels=['Không có bệnh', 'Có bệnh'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn - Tập kiểm tra với ngưỡng 0.6')
plt.show()

# Báo cáo phân loại cho tập kiểm tra với ngưỡng mới
class_report_with_threshold = classification_report(y_test, y_test_pred_with_threshold)
print('Báo cáo phân loại cho tập kiểm tra với ngưỡng 0.6:')
print(class_report_with_threshold)



