import os
import pandas as pd
import csv
import numpy as np
from data_process import get_mape, get_rmse

# parameter setting
m_window = 2  # moving window
ratio = 0.75     # train ratio
output = []

# read data
path = os.path.dirname(os.path.realpath(__file__)) + '/data/port_data.csv'
df = pd.read_csv(path, encoding='gbk')
columns = df.columns

for j in range(1, len(columns)):
    # data preprocess
    # j = 1
    print(columns[j])
    raw_series = df[columns[j]]
    raw_series = raw_series.tolist()

    train_len = int((len(raw_series) - m_window) * ratio)
    test_len = len(raw_series) - m_window - train_len

    df_x = []
    df_y = []
    for i in range(len(raw_series) - m_window):
        df_x.append(raw_series[i:i + m_window])
        df_y.append(raw_series[i + m_window])

    # forecast
    train_set = np.asarray(df_x[:train_len])
    train_fitted = train_set.mean(axis=1)

    train_real = np.asarray(df_y[:len(df_y) - test_len])

    test_set = np.asarray(df_x[train_len:])
    test_fitted = test_set.mean(axis=1)
    test_real = np.asarray(df_y[len(df_y) - test_len:])

    # calculate accuracy
    train_MAPE = get_mape(train_real, train_fitted)
    train_RMSE = get_rmse(train_real, train_fitted)
    test_MAPE = get_mape(train_real, train_fitted)
    test_RMSE = get_rmse(train_real, train_fitted)

    _output = [m_window, columns[j], train_fitted, train_real, train_MAPE, train_RMSE, test_fitted, test_real, test_MAPE, test_RMSE]
    output.append(_output)

# save result
f = open('output.csv', 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(f)
header = ('moving window', 'port', 'train_fitted', 'train_real', 'train_MAPE', 'train_RMSE', 'test_results', 'test_real', 'test_MAPE', 'test_RMSE')
csv_writer.writerow(header)
for data in output:
    csv_writer.writerow(data)
f.close()
