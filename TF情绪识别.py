
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers,regularizers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from d2l import tensorflow as d2l

def load_data():
    datasets = pd.read_csv('D:/emo.csv')

    # 将pixels列转换为numpy数组
    pixels = datasets['pixels'].to_numpy()


    # 将像素值分割为单独的行
    pixels = np.array([np.fromstring(p, sep=' ') for p in pixels])

    # 将像素值缩放到0-1之间
    pixels = pixels / 255.0

    encoder = LabelEncoder()
    emotions = encoder.fit_transform(datasets['emotion'])
    X_train, X_test, y_train, y_test = train_test_split(pixels, emotions, test_size=0.2, random_state=7)
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)
    return X_train, X_test, y_train, y_test

def build_model():

    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(7, activation='softmax')
])

    model.build(input_shape=(None, 48, 48, 1))
    model.summary()

    #optimizer = tf.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
build_model()
model=build_model()
#model.save_weights('model.tf')

X_train,X_test,y_train,y_test=load_data()
history =model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
def plt_train():
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.show()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.show()
plt_train()
