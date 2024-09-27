import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Tải dữ liệu
df = pd.read_csv('./mynewdata.csv')
df = df.drop([0, 1, 2])  # Xóa 3 dòng dữ liệu đầu để cho đủ 300 mẫu

# Tách dữ liệu và nhãn
X = df.drop(columns=['target', 'STT']) 
y = df['target']  # Nhãn mục tiêu

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

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Huấn luyện mô hình Perceptron
model = Perceptron(max_iter=2000, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Lưu mô hình
joblib.dump(model, "PLA.pkl")

# Dự đoán và đánh giá trên tập xác thực
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Độ chính xác trên tập xác thực: {val_accuracy:.10f}')

# Dự đoán và đánh giá trên tập kiểm tra
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Độ chính xác trên tập kiểm tra: {test_accuracy:.10f}')

# Ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_test_pred)
print('Ma trận nhầm lẫn:')
print(conf_matrix)

# Vẽ biểu đồ confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Ma trận nhầm lẫn")
plt.xlabel("Nhãn Dự Đoán")
plt.ylabel("Nhãn Thực")
plt.show()

# Báo cáo phân loại
class_report = classification_report(y_test, y_test_pred)
print('Báo cáo phân loại:')
print(class_report)

# Tính và vẽ ROC Curve
if len(set(y)) == 2:
    # Dự đoán xác suất
    y_test_proba = model.decision_function(X_test)  # Thay vì predict_proba với Perceptron

    # Tính toán ROC và AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    auc = roc_auc_score(y_test, y_test_proba)

    # Vẽ ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Đường tham chiếu
    plt.xlabel('Tỷ lệ Dương Giả')
    plt.ylabel('Tỷ lệ Dương Thực')
    plt.title('Đường Cong ROC')
    plt.legend(loc="lower right")
    plt.show()

    # Vẽ biểu đồ phân phối xác suất dự đoán
    plt.figure(figsize=(8, 6))
    sns.histplot(y_test_proba, bins=20, kde=True, color='blue')
    plt.title("Phân Phối Xác Suất Dự Đoán")
    plt.xlabel("Xác Suất Dự Đoán")
    plt.ylabel("Tần Suất")
    plt.show()


