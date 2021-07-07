from keras.datasets import mnist
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import keras 

(x_train,y_train),(x_test,y_test)= mnist.load_data()

x_train_ = np.expand_dims(x_train,3)
y_train_ = to_categorical(y_train)
x_test_ = np.expand_dims(x_test,3)
y_test_ = to_categorical(y_test)

def cnn():
    model = Sequential()
    model.add(Conv2D(input_shape=(28,28,1),filters=25,kernel_size=(3,3),padding='same',activation='relu')) 
    model.add(MaxPool2D())
    model.add(Dropout(rate=0.3))
    model.add(Flatten())
    model.add(Dense(32,activation='relu'))
    model.add((Dense(10,activation='softmax')))
    return model



batchsize = 64
dataset_size = x_train_.shape[0]
print(dataset_size)

def generator(x_train_,y_train_,batchsize=batchsize,shuffle=True):
    dataset_size = x_train_.shape[0]
    while True:
        i = 0 
        if shuffle:
            img_index = np.random.randint(0,dataset_size,batchsize)
            img = x_train_[img_index]
            label = y_train_[img_index]
        else:
            img = x_train_[i:((i+batchsize)% dataset_size)]
            label = y_train_[i:((i+batchsize)% dataset_size)]
            i = (i+batchsize)% dataset_size
        yield (img,label)
gen = generator(x_train_,y_train_,shuffle=True)
net = cnn()
net.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(0.001),metrics=['accuracy'])
net.fit_generator(generator=gen,steps_per_epoch=dataset_size/batchsize,epochs=5,validation_data=(x_test_,y_test_))        
        


