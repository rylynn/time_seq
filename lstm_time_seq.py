from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import sklearn
#from sklearn.preprocessing import MinMaxScale
#from sklearn.preprocessing import LabelEncode
#from sklearn.metrics import mean_squared_erro
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
# 转换序列成监督学习问题
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# 加载数据集 , raw_data.csv
def get_data(path, drop_column = []):
    dataset = read_csv(path, header=0, index_col=0)
    dataset = dataset.drop(drop_column, axis=1)
    cols=[x for i,x in enumerate(dataset.columns) if dataset.iat[0,i] =='NA' or 'None']
    dataset = dataset.drop(cols, axis=1)
    values = dataset.values
    print(values)
    return values

# 整数编码
def process_data(values):
    encoder = sklearn.preprocessing.LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])
    # 归一化特征
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # 构建监督学习问题
    reframed = series_to_supervised(scaled, 1, 1)
    # 丢弃我们并不想预测的列
    reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)

    # ensure all data is float
    values = values.astype('float32')
    print(reframed.head())

    values = reframed.values
    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # 分为输入输出
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y.g = test[:, :-1], test[:, -1]
    # 重塑成3D形状 [样例, 时间步, 特征]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    print(train_x.shape, train_y.shape, test_x.shape, test_y.g.shape)
    return train_x, train_y, test_x, test_y.g

# 设计网络
def init_network(in_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=in_shape))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

def train_model(model, train_x, train_y, test_x, test_y, save_path = 'time_seq.model'):
    # 拟合神经网络模型
    history = model.fit(train_x, train_y, epochs=50, batch_size=128, validation_data=(test_x, test_y), verbose=2, shuffle=False)
    model.save(save_path)
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Score: loss-',score[0])
    print('AUC:', score[1])
    # 绘制历史数据
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

def predict_with_model(model, values):
    ret = model.predict(values)
    print(values)
    print("result:" % ret);

if __name__ == "__main__":
    df = get_data("./raw_data.csv")   
    train_x, train_y, test_x, test_y = process_data(df)
    model = init_network((train_x.shape[1], train_x.shape[2]))   
    train_model(model, train_x, train_y, test_x, test_y)
'''
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# 反向转换预测值比例
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# 反向转换实际值比例
test_y.g = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y.g, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# 计算RMSE
rmse = sqrt(sklearn.metrics.mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
'''
