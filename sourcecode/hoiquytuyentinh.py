import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from function import NSE, R2, RMSE, MAE

data_df = pd.read_csv("../DATA_COIN.csv")

dt_Train, dt_Test = train_test_split(data_df, test_size=0.3, shuffle=False)

x_train = dt_Train.iloc[:, 1:9]

y_train = dt_Train.iloc[:, 9]

X_test = dt_Test.iloc[:, 1:9]

Y_test = dt_Test.iloc[:, 9]

reg = LinearRegression().fit(x_train, y_train)

y_pred = reg.predict(X_test)

y = np.array(Y_test)

print("Thuc te         Du Doan                 Chenh lech")

for i in range(0, len(y)):
    print("%.2f" % y[i], "    ", y_pred[i], "    ", abs(y[i] - y_pred[i]))

print('\n')

print('Hệ số xác định theo NSE là: %.2f' % abs(NSE(Y_test, y_pred)))

print("Hệ số xác định theo R2 là %.2f" % R2(Y_test, y_pred))

print("Hệ số xác định theo MAE là %.2f" % MAE(Y_test, y_pred))

print("Hệ số xác định theo RMSE là %.2f" % RMSE(Y_test, y_pred))
