import sys
from keras import applications
import numpy as np
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import keras
from keras.models import Model
from keras.engine.topology import Layer, InputSpec
import keras.utils.conv_utils as conv_utils
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate
#base_model = applications.DenseNet169(input_shape=(224, 224, 3), include_top=True)
#densenet如果需要更改input_shape，需要include_top = False，也就是是否包含dense层，也就是全连接层
base_model = applications.DenseNet169(input_shape=(None, None, 3), include_top=False)
 
#base_model.summary()#模型列表
#print(base_model.output)
#Tensor("relu/Relu:0", shape=(?, ?, ?, 1664), dtype=float32)

base_model_output_shape = base_model.layers[-1].output.shape
#print(base_model_output_shape)#(?, ?, ?, 1664)
decode_filters = int(int(base_model_output_shape[-1])/2)
#print(base_model_output_shape[-1])#1664
#print(decode_filters)#832
print('the number of layers in this model:'+str(len(base_model.layers)))#595


def upproject(tensor, filters, name, concat_with):
    up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
    up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
    up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
    up_i = LeakyReLU(alpha=0.2)(up_i)
    up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
    up_i = LeakyReLU(alpha=0.2)(up_i)
    return up_i
class BilinearUpSampling2D(Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        
        return tf.image.resize(inputs, [height, width], method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def normalize_data_format(value):
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format
decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.output)
#print(decoder)
#Tensor("conv2/BiasAdd:0", shape=(?, ?, ?, 832), dtype=float32)
decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
#print(decoder)
#Tensor("leaky_re_lu_2/LeakyRelu:0", shape=(?, ?, ?, 416), dtype=float32)
decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
#print(decoder)
#Tensor("leaky_re_lu_4/LeakyRelu:0", shape=(?, ?, ?, 208), dtype=float32)
decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
#print(decoder)
#Tensor("leaky_re_lu_6/LeakyRelu:0", shape=(?, ?, ?, 104), dtype=float32)
decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
#print(decoder)
#Tensor("leaky_re_lu_8/LeakyRelu:0", shape=(?, ?, ?, 52), dtype=float32)

conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)
#print(conv3)
#Tensor("conv3/BiasAdd:0", shape=(?, ?, ?, 1), dtype=float32)

model = Model(inputs=base_model.input, outputs=conv3)
print(model)
model.summary()#模型列表
print('the number of layers in this model:'+str(len(model.layers)))#595