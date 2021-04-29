# README

## 1. 环境

可以使用anaconda，创建一个虚拟环境。使用的python版本是3.6

```
conda create -n temp_link_pred python=3.6
pip install tensorflow==1.14.0
pip install keras==2.2.4
pip install xgboost
pip install pandas
pip install skelearn
pip install statsmodels
pip install matplotlib
```



## 2. 数据

### "address.csv": 15个以太坊交易所的地址

| 地址                                       | 对应交易所 |
| ------------------------------------------ | ---------- |
| 0x0681d8db095565fe8a346fa0277bffde9c0edbbf | Binance    |
| 0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be | Binance    |
| 0xd551234ae421e3bcba99a0da6d736074f22192ff | Binance    |
| 0x0577a79cfc63bbc0df38833ff4c4a3bf2095b404 | Huobi      |
| 0x28ffe35688ffffd0659aee2e34778b0ae4e193ad | Huobi      |
| 0x3c979fb790c86e361738ed17588c1e8b4c4cc49a | Huobi      |
| 0x58c2cb4a6bee98c309215d0d2a38d7f8aa71211c | Huobi      |
| 0x73f8fc2e74302eb2efda125a326655acf0dc2d1b | Huobi      |
| 0x794d28ac31bcb136294761a556b68d2634094153 | Huobi      |
| 0xc9610be2843f1618edfedd0860dc43551c727061 | Huobi      |
| 0xf66852bc122fd40bfecc63cd48217e88bda12109 | Huobi      |
| 0x2b5634c42055806a59e9107ed44d43c426e58258 | KuCoin     |
| 0x689c56aef474df92d44a1b70850f808488f9769c | KuCoin     |
| 0x0211f3cedbef3143223d3acf0e589747933e8527 | MXC        |
| 0x75e89d5979e4f6fba9f97c104c2f0afb3f1dcb88 | MXC        |



### "./data/source_data"

原始数据，15个地址的所有交易记录，时间段为2020年6月至10月，处理特征选取时间段为2020年6月22日到2020年9月9日的80天。

### "./data/feature_4_1"

由原始数据处理出来的数据，4表示4个特征，分别是交易次数、交易总额、交易额均值、交易额方差，1表示time step为1天。

"./data/feature_4_3"表示timestep为3天的特征，后面实验是基于前者，所以这个没有用

### "arima_4_1", "GRU_4_1", "LSTM_4_1", "RNN_4_1", "randomforest_4_1", "xgboost_4_1", "svr_4_1", "LA_HA_4_1"

基于特征文件"./data/feature_4_1"的各种方法的预测结果

### "./figure"

文件夹里面是各种方法预测方法的预测曲线图，每个图是每个地址的预测曲线

### "./n_units.eps", "./n_window.eps"

参数敏感性曲线，"_"后面加方法

## 3. 代码

### "./feature_extraction.py"

由原始数据抽取特征。

```python
# 打开address文件，遍历每个地址的交易记录
file = open('./address.csv')
df = pd.read_csv(file)
address = df['address']

# 抽取特征
def feature_extraction(ts = 80, minTime=1590940800, n=1):
    # 80表示总共80天，1590940800是2020年6月1日的时间戳，1是timestp为1天
    
    # 创建特征文件(csv)
    for i in range(len(address)):
        #如果不存在该路径
        if not os.path.exists('./data/feature_4_{}'.format(n)):
            os.makedirs('./data/feature_4_{}'.format(n))
        file2 = open('./data/feature_4_{}/{}_ft.csv'.format(n, address[i]), 'w',
                    newline='')
        csvwriter = csv.writer(file2)
        #四个特征
        csvwriter.writerow(['tran_num', 'tran_sum', 'tran_mean', 'tran_var'])

        for j in range(ts):
            csvwriter.writerow([0. for i in range(4)])
        file2.close()
        
    # 遍历每个地址的原始文件
    for i in range(len(address)):
        print('i={}'.format(i))
        node = address[i]
        data = open('./data/source_data/{}.csv'.format(node))
        df_data = pd.read_csv(data)

        # 用一个二维list保存每个地址的80天（ts）的交易，每天用一个list存储，可以进一步求均值，方差，次数等
        tran = [[] for i in range(ts)]

        for j in range(len(df_data)):
            ft = open('./data/feature_4_{}/{}_ft.csv'.format(n, address[i]))
            df_ft = pd.read_csv(ft)
            ft.close()
			
            # 将时间戳转化为天数
            t = (df_data['TimeStamp'][j] - minTime) // (86400 * n)
            # 因为选择的时间段是2020年6月1日起的第21天到100天，即21-80变成0-79
            if n==1:
                t-=21
            elif n==3:
                t-=7
            # 如果交易发生在我们感兴趣的时间段内，储存在list里
            if t>=0 and t<ts :
                
                df_ft['tran_num'][t] = df_ft['tran_num'][t] + 1
                df_ft['tran_sum'][t] = df_ft['tran_sum'][t] + df_data['Value'][j]

                tran[t].append(df_data['Value'][j])

            df_ft.to_csv('./data/feature_4_{}/{}_ft.csv'.format(n, address[i]),
                         index=False)
        for t in range(ts):
            if len(tran[t])>0:
                df_ft['tran_mean'][t] = np.mean(tran[t])
                df_ft['tran_var'][t] = np.var(tran[t])

            else:
                df_ft['tran_mean'][t] = 0
                df_ft['tran_var'][t] = 0

        df_ft.to_csv('./data/feature_4_{}/{}_ft.csv'.format(n, address[i]),
                     index=False)
```



### "./temporal.py"

#### 时序注意力

```python
class attention(Layer):
```

#### LSTM

可以通过methods修改为gru或者rnn

```python
def lstm(n_features=4, #多少个特征
         n_train=60,   #训练集长度
         n_window=6,   #滑动窗口的大小
         n_units=64,   #lstm，gru，rnn 的units
         n_epochs=10,  #训练轮次
         with_att=False, #是否加入注意力
         methods='lstm', #选择的方法
         lr=0.001,     #学习率
         n_gap = 1,    #timestep 1天
         feature_n = 1 #比较第几个特征，1即交易总额
         ):
    # 读取所有特征文件
    data = []

    for i in range(len(address)):
        f = open('./data/feature_{}_{}/{}_ft.csv'.format(n_features, n_gap, address[i]))
        df = pd.read_csv(f)
        data.append(df.values)
        
    data = np.array(data)
    
    # 标准化数据，维度变化
    scaler = MinMaxScaler(feature_range=(0, 1))
    n_samples, n_timesteps, n_features = data.shape
    scaled_data = data.reshape((n_samples, n_timesteps*n_features))
    scaled_data = scaler.fit_transform(scaled_data)
    scaled_data = scaled_data.reshape((n_samples, n_timesteps, n_features))
    
    # 训练集长度、测试集长度
    n_test = n_timesteps - n_train
    
    # 模型构建
    inputs = Input(shape=(n_window, n_features))
    return_sequences = False
    # 是否加入注意力
    if with_att==True:
        return_sequences = True
    # 选择方法
    if methods=='lstm':
        att_in = Bidirectional(LSTM(n_units, input_shape=(n_window, n_features), return_sequences=return_sequences))(inputs)
    elif methods=='gru':
        att_in = Bidirectional(GRU(n_units, input_shape=(n_window, n_features), return_sequences=return_sequences))(inputs)
    elif methods=='rnn':
        att_in = Bidirectional(SimpleRNN(n_units, input_shape=(n_window, n_features), return_sequences=return_sequences))(inputs)
    if with_att==True:
        att_out = attention()(att_in)
        outputs = Dense(1)(att_out)
    else:
        outputs = Dense(1)(att_in)

    model = Model(inputs, outputs)
    opt = optimizers.Adam(lr=lr)
    model.compile(loss='mse', optimizer=opt)
    
    #拟合模型
    for i in range(n_train-n_window):
    history = model.fit(scaled_data[:, i: i+n_window, :], scaled_data[:, i+n_window, feature_n], epochs=n_epochs)
    
    # 预测
    inv_yhat = []
    for i in range(n_test):
        yhat = model.predict(scaled_data[:, n_train-n_window+i:n_train+i, :])
        inv_yhat.append(yhat)
	# 维度变化
    inv_yhat = np.array(inv_yhat)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape)) #inv_yhat.shape:(20, 15, 1)
    inv_yhat = inv_yhat.reshape((inv_yhat.shape[0], inv_yhat.shape[1]))
    print('inv_yhat.shape:{}'.format(inv_yhat.shape)) #inv_yhat.shape:(20, 15)
    inv_yhat = inv_yhat.T
    print('inv_yhat.shape:{}'.format(inv_yhat.shape)) #inv_yhat.shape:(15, 20)
    
    # 这一段是选择比较哪一个特征，我们选择特征1，即feature_n = 1，交易总额。
    temp = scaled_data[:, n_train:, 0]
    temp = temp.reshape(temp.shape[0], temp.shape[1], 1)
    for k in range(1, feature_n):
        temp_ = scaled_data[:, n_train:, k]
        temp_ = temp_.reshape(temp_.shape[0], temp_.shape[1], 1)
        temp = np.concatenate((temp, temp_), axis=2)
    inv_yhat = inv_yhat.reshape(inv_yhat.shape[0], inv_yhat.shape[1], 1)
    inv_yhat = np.concatenate((temp, inv_yhat), axis=2)
    for k in range(feature_n+1, n_features):
        temp_ = scaled_data[:, n_train:, k]
        temp_ = temp_.reshape(temp_.shape[0], temp_.shape[1], 1)
        inv_yhat = np.concatenate((inv_yhat, temp_), axis=2)
    print('inv_yhat.shape1:{}'.format(inv_yhat.shape))
    print(inv_yhat)
    inv_yhat = inv_yhat.reshape((n_samples, n_test, n_features))
    print('inv_yhat.shape2:{}'.format(inv_yhat.shape))
    print(inv_yhat)
    inv_yhat = np.concatenate((scaled_data[:, :n_train, :], inv_yhat), axis=1)
    print('hhhhh={}'.format(inv_yhat.shape))

    print('inv_yhat.shape:{}'.format(inv_yhat.shape))
    inv_yhat = inv_yhat.reshape(n_samples, n_timesteps*n_features)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat.reshape(n_samples, n_timesteps, n_features)
    # 小于零的预测值转为0
    inv_yhat[inv_yhat<0] = 0 # transform negative values to zero
    prediction = inv_yhat[:, -n_test:, feature_n]
    prediction = prediction.reshape(prediction.shape[0], prediction.shape[1], 1)
    original = data[:, -n_test:, feature_n]
    original = original.reshape(original.shape[0], original.shape[1], 1)
    concat = np.concatenate((original, prediction), axis=2)
    print('concat.shape:{}'.format(concat.shape))
    np.set_printoptions(threshold=1e6)
    print('concat\n{}'.format(concat))
    concat = concat.reshape(concat.shape[0]*concat.shape[1], concat.shape[2])
    
    # 保存预测
    df = pd.DataFrame(concat)
    df.columns = ['original', 'prediction']
    if not os.path.exists('./data/{}_{}_{}'.format(methods.upper(), n_features, n_gap)):
        os.makedirs('./data/{}_{}_{}'.format(methods.upper(), n_features, n_gap))
    df.to_csv('./data/{}_{}_{}/prediction_{}.csv'.format(methods.upper(), n_features, n_gap, methods.upper(),), index=False)
    
    # 各种评判标准
    rmse = sqrt(mean_squared_error(df['original'].values, df['prediction'].values))
    mae = mean_absolute_error(df['original'].values, df['prediction'].values)
    mape = mean_absolute_percentage_error(df['original'].values, df['prediction'].values)
    r2 = r2_score(df['original'].values, df['prediction'].values)
    print('rmse: {}'.format(rmse))
    print('mae: {}'.format(mae))
    print('mape: {}'.format(mape))
    print('r2: {}'.format(r2))
    return rmse, mae, mape, r2
```

#### LA和HA

```python
def la_ha(n_train=60,
          n_features=4,
          n_gap=1,
          n_timestamp=80
          ):
```

#### arima

```python
def arima(n_train=60, p=2, d=1, q=2, n_features=4, n_gap=1):
```

#### xgboost、arima

```python
#数据处理
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    
#划分训练集、测试集
def train_test_split(data, n_test):
    
#根据methods选择使用rf还是xgboost进行预测
#调用random_forest_forecast 和 xgboost_forecast
def walk_forward_validation(data, n_test, methods='randomforest'):
    
#拟合模型，进行预测
def random_forest_forecast(train, testX):
    
def xgboost_forecast(train, testX):

#保存预测文件，进行评估
def randomforest(n_test=20, n_features=4, n_gap=1):

def xgboost(n_test=20, n_features=4, n_gap=1):


    
```



#### svr

```python
#数据预处理，划分数据集为训练集和测试集
def preprocess_data_svr(values, train_size=60, time_len=80, 
                        seq_len=6,#滑动窗口长度
                        pre_len=1):
    
def svr(n_train=60, n_test=20, n_window=6, n_features=4, n_gap=1):
```

#### 评判标准

```python
def evaluation(real, pre):
    rmse = mean_squared_error(real, pre, squared=False)
    mae = mean_absolute_error(real, pre)
    mape = mean_absolute_percentage_error(real, pre)
    r2 = r2_score(real, pre)
    return rmse, mae, mape, r2
```

#### 画各种方法的预测曲线

```python
def plot_curve(n_gap=1, n_features=4, n_train=60, n_timestamp=80):
```



### "./temporal_spatial.py"

待完成。。。