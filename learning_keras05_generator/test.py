# #######################################banch_size,epoch,banch,Iterations的概念##################################
# 在训练模型时，如果训练数据过多，无法一次性将所有数据送入计算，那么我们就会遇到epoch，batchsize，iterations这些概念。
# 为了克服数据量多的问题，我们会选择将数据分成几个部分，即batch，进行训练，从而使得每个批次的数据量是可以负载的。将这些batch的数据逐一送入计算训练，更新神经网络的权值，使得网络收敛。
# 一个epoch指代所有的数据送入网络中完成一次前向计算及反向传播的过程。由于一个epoch常常太大，计算机无法负荷，我们会将它分成几个较小的batches。
# Batch(单个batches)就是每次送入网络中训练的一部分数据，而Batch_Size就是每次送入网络中训练样本的数量
# iterations就是完成一次epoch所需的batch个数。刚刚提到的，batches的数量就是iterations。
# 训练有2000个数据，分成4个batch，所以完成1个epoch需要进行4次iterations。那么每个batch里的数据就是2000/4，所以batch_size就是500。
# #################################################

#################为什么要使用 fit_generator######################
# 在进行神经网络训练的时候如果使用 model.fit 的方式训练，需要把整个 x_train 加载到内存中，
# 而 keras 自带的一些 datasets 又很小，在训练的时候完全可以这么做，所以如果是 keras 的小白，可能觉得 fit 是一个很好用的训练方法，而且也习惯了这种方式。
# 但是，如果我现在想用 ImageNet 数据集（1000000张图片，1000个类别）来进行训练，直接加载到内存，内存会爆掉，所以我们要批量地把他们放到内存里。
# 那么这就产生了一个问题，我们使用 fit 来训练的时候，在接口中封装了 batch_size 的参数，我们只需要输入一个数字，他就能够按照我们输入的尺寸对数据集进行分割，方便又快捷，
# 那么如果我们手动地让他们一批批地送入内存中进行训练，该如何实现呢？创建一个 generator 来替你完成原来 fit 中 batch_size 所做的工作
###########################################
import keras
from keras.models import Sequential
from keras.utils import Sequence
from augment import BasicPolicy
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
import cv2
marked = True
batch_size = 4
kernel_sharpen = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)#解压数据集
    return {name: input_zip.read(name) for name in input_zip.namelist()}#将数据集读取为字典
data = extract_zip("/home/lzq/DenseDepth-master/nyu_data.zip")

# ##############################################查看数据集的存储格式#################################################
# print(type(data))
# #<class 'dict'>
# print("Length : %d" % len (data))
# #Length : 102973
# a = data.keys()#取出字典的key
# print(type(a))#查看key的数据类型
# #<class 'dict_keys'>
# print(list(a)[0:20])#将dict_keys转化成list，输出前21项
# #['data/', 'data/nyu2_test/', 'data/nyu2_test.csv', 'data/nyu2_test/00000_colors.png', 'data/nyu2_test/00000_depth.png', 'data/nyu2_test/00001_colors.png', 
# # 'data/nyu2_test/00001_depth.png', 'data/nyu2_test/00008_colors.png', 'data/nyu2_test/00008_depth.png', 'data/nyu2_test/00013_colors.png', 
# # 'data/nyu2_test/00013_depth.png', 'data/nyu2_test/00014_colors.png', 'data/nyu2_test/00014_depth.png', 'data/nyu2_test/00015_colors.png',
# #  'data/nyu2_test/00015_depth.png', 'data/nyu2_test/00016_colors.png', 'data/nyu2_test/00016_depth.png', 'data/nyu2_test/00017_colors.png',
# #  'data/nyu2_test/00017_depth.png', 'data/nyu2_test/00020_colors.png']


# print ("Value : %s" %  data.get('data/')) #查看字典中第一二项key对应的value
# #Value : b''
# print ("Value : %s" %  data.get('data/nyu2_test/')) 
# #Value : b''

# train_color_0 = data.get('data/nyu2_train/basement_0001a_out/1.jpg')#取出训练集中的一个rgb图的key对应的value
# train_depth_0 = data.get('data/nyu2_train/basement_0001a_out/1.png')#取出训练集中的一个深度图的key对应的value
# print(type(train_color_0))#查看图片的数据类型
# #<class 'bytes'>
# print(type(train_depth_0))
# #<class 'bytes'>

# #将rgb图片数据从bytes转为asarray
# train_color_0 = np.asarray(Image.open( BytesIO(train_color_0)))
# print(type(train_color_0))#查看转换后rgb图片的数据类型
# #<class 'numpy.ndarray'>
# print(train_color_0.shape)#查看转化后rgb图片的shape
# #(480, 640, 3)
# print(train_color_0[100][100][2])#输出此rgb图片在[100][100][2]处的值
# #6

# ############图像锐化
# cv2.imshow('Image',train_color_0)
# train_color_0 = cv2.filter2D(train_color_0,-1,kernel_sharpen)
# cv2.imshow('H1_sharpen Image',train_color_0)
# if cv2.waitKey(0) & 0xFF == 27:
#     cv2.destroyAllWindows()



# #rgb图片预处理
# train_color_0 = np.clip(train_color_0.reshape(480,640,3)/255,0,1)
# #####clip这个函数将将数组中的元素限制在a_min, a_max之间,大于a_max的就使得它等于 a_max,小于a_min,的就使得它等于a_min。
# print(type(train_color_0))#查看预处理后rgb图片的数据类型
# #<class 'numpy.ndarray'>
# print(train_color_0.shape)#查看预处理后rgb图片的shape
# #(480, 640, 3)
# print(train_color_0[100][100][2])#输出预处理后此rgb图片在[100][100][2]处的值
# #0.023529411764705882

# #将深度图片数据从bytes转为asarray
# train_depth_0 = np.asarray(Image.open( BytesIO(train_depth_0)))
# print(type(train_depth_0))#查看转换后深度图图片的数据类型
# #<class 'numpy.ndarray'>
# print(train_depth_0.shape)#查看转化后深度图片的shape
# # (480, 640)
# print(train_depth_0[100][100])#输出此深度图在[100][100]处的值
# # 110

# #深度图片预处理
# train_depth_0 = np.clip(train_depth_0.reshape(480,640,1)/255*1000.0,0,1000.0)
# print(type(train_depth_0))#查看预处理后深度图图片的数据类型
# #<class 'numpy.ndarray'>
# print(train_depth_0.shape)#查看预处理后深度图片的shape
# # (480, 640, 1)
# print(train_depth_0[100][100])#输出此深度图预处理后在[100][100]处的值
# # [431.37254902]

# # # #######################################################################################################################################
#将训练集和测试集里的color和depth成对放到list里
nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))
# print(type(nyu2_train))
# #<class 'list'>
# print(type(nyu2_test))
# #<class 'list'>
# print(nyu2_train[1])#输出训练集list里的【1】个元素
# #['data/nyu2_train/living_room_0038_out/115.jpg', 'data/nyu2_train/living_room_0038_out/115.png']
# print(nyu2_test[1])
# #['data/nyu2_test/00001_colors.png', 'data/nyu2_test/00001_depth.png']
shape_rgb = (batch_size, 480, 640, 3)
shape_depth = (batch_size, 240, 320, 1)

class NYU_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Augmentation of RGB images   
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]
# ######################
#             x = Image.open( BytesIO(self.data[sample[0]]))
#             x = cv2.filter2D(x,-1,kernel_sharpen)
#             x = np.clip(np.asarray(x.reshape(480,640,3)/255,0,1)) 
# ###############################
            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]) )).reshape(480,640,3)/255,0,1)
            x = cv2.filter2D(x,-1,kernel_sharpen)
            y = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[1]]) )).reshape(480,640,1)/255*self.maxDepth,0,self.maxDepth)
            global marked
            if marked:
                print(x.shape)
                marked = False


            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y
train_generator = NYU_BasicAugmentRGBSequence(data, nyu2_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
print(type(train_generator))
train_generator.__getitem__(1)
