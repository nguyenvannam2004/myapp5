import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score,classification_report

# Hàm kích hoạt Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Đạo hàm của hàm kích hoạt Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Bước 1: Đọc dữ liệu bệnh tim từ file CSV 
data = pd.read_csv('./mynewdata.csv')

# Giả sử các đặc trưng (feature) của bạn là các cột từ 0 đến -1 và cột đích là 'target'
X = data.drop(columns=['target', 'STT']).values  # Các đặc trưng đầu vào
y = data['target'].values  # Nhãn mục tiêu

# Chuẩn hóa dữ liệu
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

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


# Khởi tạo trọng số ngẫu nhiên
np.random.seed(42)
input_size = X_train.shape[1]  # Kích thước đầu vào
hidden_size = 3  # Số lượng nơ-ron trong lớp ẩn
output_size = 1  # Số lượng nơ-ron đầu ra

# Trọng số từ đầu vào đến lớp ẩn
weights_input_hidden = np.random.rand(input_size, hidden_size)  
# Trọng số từ lớp ẩn đến đầu ra
weights_hidden_output = np.random.rand(hidden_size, output_size)  

# Tham số
learning_rate = 0.01 # Thay đổi learning rate
epochs = 2000  # Giảm số lượng epochs để thử nghiệm

# Tham số cho regularization
lambda_reg = 0.02  # Tham số điều chỉnh cho regularization

# Huấn luyện mạng nơ-ron
for epoch in range(epochs):
    # Giai đoạn lan truyền tiến
    hidden_layer_input = np.dot(X_train, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    final_input = np.dot(hidden_layer_output, weights_hidden_output)
    final_output = sigmoid(final_input)
    
    # Tính toán lỗi
    error = y_train.reshape(-1, 1) - final_output

    # Lan truyền ngược
    d_final_output = error * sigmoid_derivative(final_output)
    error_hidden_layer = d_final_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Cập nhật trọng số với regularization
    weights_hidden_output += hidden_layer_output.T.dot(d_final_output) * learning_rate - lambda_reg * weights_hidden_output
    weights_input_hidden += X_train.T.dot(d_hidden_layer) * learning_rate - lambda_reg * weights_input_hidden

# Dự đoán trên tập xác thực
hidden_layer_input_val = np.dot(X_val, weights_input_hidden)
hidden_layer_output_val = sigmoid(hidden_layer_input_val)

final_input_val = np.dot(hidden_layer_output_val, weights_hidden_output)
final_output_val = sigmoid(final_input_val)

# Chuyển đổi đầu ra thành nhãn dự đoán
y_val_pred = (final_output_val > 0.5).astype(int)

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Dộ chính xác trên tập xác thực: {val_accuracy:.10f}')

# Dự đoán và đánh giá trên tập kiểm tra
hidden_layer_input_test = np.dot(X_test, weights_input_hidden)
hidden_layer_output_test = sigmoid(hidden_layer_input_test)

final_input_test = np.dot(hidden_layer_output_test, weights_hidden_output)
final_output_test = sigmoid(final_input_test)

# Chuyển đổi đầu ra thành nhãn dự đoán
y_test_pred = (final_output_test > 0.5).astype(int)

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Dộ chính xác trên tập kiểm tra: {test_accuracy:.10f}')

# Tính toán xác suất dự đoán
y_test_proba = final_output_test.flatten()  

# Tính và vẽ ROC Curve (chỉ áp dụng với bài toán nhị phân)
if len(set(y)) == 2:
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

new_data = np.array([61,1,0,120,260,0,1,140,1,3.6,1,1,3])

def predict(X_new):
    global weights_input_hidden, weights_hidden_output
    # Chuẩn hóa dữ liệu mới
    X_new = (X_new - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Giai đoạn lan truyền tiến
    hidden_layer_input = np.dot(X_new, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    final_input = np.dot(hidden_layer_output, weights_hidden_output)
    final_output = sigmoid(final_input)

    # Chuyển đổi đầu ra thành nhãn dự đoán
    y_pred = (final_output > 0.99).astype(int)
    return y_pred

# Ví dụ về cách sử dụng hàm dự đoán
# Giả sử bạn có dữ liệu mới cần dự đoán
predictions = predict(new_data)


# In kết quả dự đoán và xác suất
print("Dự đoán:", predictions)


# Lưu trọng số của mô hình vào file
model = {
    "weights_input_hidden": weights_input_hidden,
    "weights_hidden_output": weights_hidden_output
}
joblib.dump(model, "neuralnetwork.pkl")


