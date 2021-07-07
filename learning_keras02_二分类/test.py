from keras.models import Sequential
from keras.layers import Dense, Activation
#模型搭建阶段
model= Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
# Dense(32) 是一个有32个隐藏单元的完全连接层
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 生成虚拟数据
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))
print(labels)

# Train the model, iterating on the data in batches of 32 samples 训练模型，成批迭代32个样本的数据
model.fit(data, labels, nb_epoch =5, batch_size=32)

