import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from function import NSE, MAE, RMSE, R2

# Loading the dataset
data = pd.read_csv('../DATA_COIN.csv')
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)
# Implementing cross validation
k = 3
kf = KFold(n_splits=k, random_state=None)


# tinh error, y thuc te, y_pred: dl du doan
def error(y, y_pred):  # tính tổng sai lệch rồi chia trung binh, định nghĩa hàm lỗi
    sum = 0
    for i in range(0, len(y)):
        sum = sum + abs(y[i] - y_pred[i])
    return sum / len(y)  # tra ve trung binh


min = 999999
for train_index, validation_index in kf.split(dt_Train):  #
    X_train, X_validation = dt_Train.iloc[train_index, 1:9], dt_Train.iloc[validation_index, 1:9]
    y_train, y_validation = dt_Train.iloc[train_index, 9], dt_Train.iloc[validation_index, 9]  # lấy cột thứ 9
    lr = LinearRegression()  # gọi mô hình hồi quy tuyến tính
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_validation_pred = lr.predict(X_validation)  # truyền vào để dự đoán
    y_train = np.array(y_train)
    y_validation = np.array(y_validation)
    sum_error = error(y_train, y_train_pred) + error(y_validation,
                                                     y_validation_pred)  # so sánh thực tế và dự đoán trên tệp train
    if sum_error < min:  # kiểm tra nếu nhỏ hơn thì sẽ lưu lỗi min và lưu lại mô hình
        min = sum_error
        regr = lr
y_test_pred = regr.predict(dt_Test.iloc[:, 1:9])  # dự đoán kết quả trên tập test lấy 9 cột đầu tiên
y_test = np.array(dt_Test.iloc[:, 9])  # đọc cột  thứ 9

print("Thuc te         Du doan                Chenh lech")
for i in range(0, len(y_test)):
    print(y_test[i], "     ", y_test_pred[i], "     ", abs(y_test[i] - y_test_pred[i]))

print('\n')
#
print('Hệ số xác định theo NSE là: %.2f' % abs(NSE(y_test, y_test_pred)))

#
print("Hệ số xác định theo R2 là %.2f" % R2(y_test, y_test_pred))

# Đánh giá mô hình bằng cách tính trung bình giá trị tuyệt đối sai số giữa giá trị thực tế và giá trị dự đoán
print("Hệ số xác định theo MAE là %.2f" % MAE(y_test, y_test_pred))

# Tính phương trình phương sai giữa giá trị thực tế và giá trị dự đoán
print("Hệ số xác định theo RMSE là %.2f" % RMSE(y_test, y_test_pred))
