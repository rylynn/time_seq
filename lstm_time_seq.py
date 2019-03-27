from math import sqrt
from math import floor
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn import preprocessing
#import MinMaxScale
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#import LabelEncode
from sklearn import metrics
#import mean_squared_erro
from keras.models import Sequential, load_model
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
    #print(agg)

    return agg
 
# 加载数据集 , raw_data.csv
def get_data(path, drop_column = []):
    dataset = read_csv(path, header=0, index_col=0)
    if len(drop_column) > 0:
        dataset = dataset.drop(drop_column, axis=1)
    cols=[x for i,x in enumerate(dataset.columns) if (dataset.iat[0,i] =='NA' or dataset.iat[0,i] == 0)]
    dataset = dataset.drop(cols, axis=1)
    values = dataset.values
    #print(values)
    return values

# 整数编码
def process_data(values):
    encoder = preprocessing.LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])
    # 归一化特征
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # 构建监督学习问题
    reframed = series_to_supervised(scaled, 1, 1)

    # ensure all data is float
    values = values.astype('float32')
    print(reframed.head())

    values = reframed.values
    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # 分为输入输出
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    # 重塑成3D形状 [样例, 时间步, 特征]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    print("train data shape ==========")
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y

def process_data_tpg(values):
    encoder = preprocessing.LabelEncoder()
    values[:,0] = encoder.fit_transform(values[:,0])
    # 归一化特征
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 500))
    #print(values[:, 4])
    scaled = scaler.fit_transform(values)
    values = scaler.fit_transform(values)
    # 构建监督学习问题
    reframed = series_to_supervised(scaled, 1, 1)

    # ensure all data is float
    #values = values[:,3:-1].astype('float32')
    #print(reframed.head())

    values = reframed.values
    #train_line = len(values[:,0]) - floor(len(values[:,0])/10)
    train_x, test_x, train_y, test_y = train_test_split(values[:,1:], values[:, 0],test_size=0.15, random_state = 42 )
    #train = values[:train_line, :]
    #test = values[train_line:, :]
    # 分为输入输出
    #train_x, train_y = train[:, 1:], train[:,0]
    #test_x, test_y = test[:, 1:], test[:, 0]
    # 重塑成3D形状 [样例, 时间步, 特征]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    print (train_x, train_y, test_x, test_y)
    return train_x, train_y, test_x, test_y

# 设计网络
def init_network(in_shape, load_path = 'time_seq.model'):
    try:
        model = load_model(load_path)
        print('load old model')
        return model
    except:
        model = Sequential()
        model.add(LSTM(50, input_shape=in_shape))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        return model

def train_model(model, train_x, train_y, test_x, test_y, save_path):
    # 拟合神经网络模型
    history = model.fit(train_x, train_y, epochs=100, batch_size=50, validation_data=(test_x, test_y), verbose=2, shuffle=False)
    model.save(save_path)
    score = model.evaluate(test_x, test_y, verbose=2)
    print('Score: loss-%f'%score)
    #print('AUC:%f'%score[1])
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
    '''
    df = get_data("./raw_data.csv", ['cbwd', 'Iws', 'Is', 'Ir'])   
    train_x, train_y, test_x, test_y = process_data(df)
    '''
    df = get_data("./effect_data.csv", ['ds', 'media'])
    model = init_network((1,2), 'time_seq_new.model')
    for i in range(5):
        train_x, train_y, test_x, test_y = process_data_tpg(df)
        print("train_x shape1:%d train_y shape1:%d", train_x.shape[1], train_x.shape[1])
        train_model(model, train_x, train_y, test_x, test_y, 'time_seq_new.model')