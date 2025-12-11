from datetime import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv('D:/NY.csv')
print(df.head())
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['date_num'] = (df['date'] - df['date'].min()).dt.days
x = df[['date_num', 'minimum temperature']]
y = df['maximum temperature']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
model = tf.keras.Sequential([
   tf.keras.layers.SimpleRNN(50, activation='relu', input_shape=(2, 1)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])

#model = tf.keras.Sequential()
# 隐含层1设置16层，权重初始化方法设置为随机高斯分布
# 加入正则化惩罚项
model.add(tf.keras.layers.Dense(16,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(32,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(1,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# ==2== 指定优化器
model.compile(optimizer=tf.keras.optimizers.SGD(0.001),loss='mean_squared_error')
# ==3== 网络训练
model.fit(x_train,y_train,validation_split=0.25,epochs=100,batch_size=128)
# ==4== 网络模型结构
model.summary()

#model.compile(optimizer='adam', loss='mse')
x_train = np.array(x_train).reshape(-1, 2, 1)
y_train = np.array(y_train).reshape(-1, 1)

x_test = np.array(x_test).reshape(-1, 2, 1)
y_test = np.array(y_test).reshape(-1, 1)

history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
predict_date = '01-04-2017'

predict_date_obj = datetime.strptime(predict_date, '%d-%m-%Y')
predict_date_num = (predict_date_obj - predict_date_obj).days
predict_input = np.array([[predict_date_num, 12]])
predicted_temperature = model.predict(predict_input)
print("预测的未来一天的最高气温为：", predicted_temperature[0][0])