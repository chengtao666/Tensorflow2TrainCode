import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# 使用keras建模方法
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

filepath = ('D:\\NY.csv')
NY=pd.read_csv(filepath)
pd
# 打印转换后的结果
print('date')
















#（6）标准化处理
from sklearn import preprocessing
input_features = preprocessing.StandardScaler().fit_transform(features)

#（7）keras构建网络模型
# ==1== 构建层次
model = tf.keras.Sequential()
# 隐含层1设置16层，权重初始化方法设置为随机高斯分布
# 加入正则化惩罚项
model.add(layers.Dense(16,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(layers.Dense(32,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(layers.Dense(1,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# ==2== 指定优化器
model.compile(optimizer=tf.keras.optimizers.SGD(0.001),loss='mean_squared_error')
# ==3== 网络训练
model.fit(input_features,targets,validation_split=0.25,epochs=100,batch_size=128)
# ==4== 网络模型结构
model.summary()
# ==5== 预测模型结果
predict = model.predict(input_features)
