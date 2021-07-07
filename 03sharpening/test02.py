# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 14:35:33 2018
1.考虑降噪、模糊、锐化怎么结合使用
2.考虑要不要在数据集上应用，是在color上应用还是在depth上应用
高斯高通滤波

cv2.bilateralFilter 双边滤波
双边滤波是一种非线性的滤波方法，是结合图像的空间邻近度和像素值相似度的一种折衷处理，同时考虑空间与信息和灰度相似性，达到保边去噪的目的，具有简单、非迭代、局部处理的特点。之所以能够达到保边去噪的滤波效果是因为滤波器由两个函数构成：一个函数是由几何空间距离决定滤波器系数，另一个是由像素差值决定滤波器系数.
cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) → dst
src：输入图像
d：过滤时周围每个像素领域的直径
sigmaColor：在color space中过滤sigma。参数越大，临近像素将会在越远的地方mix。
sigmaSpace：在coordinate space中过滤sigma。参数越大，那些颜色足够相近的的颜色的影响越大。
"""
import cv2
import numpy as np
#加载图像
#image = cv2.imread('/home/lzq/nyu_data/train/nyu_data/data/nyu2_train/study_room_0005b_out/89.png')
#image = cv2.imread('/home/lzq/nyu_data/train/nyu_data/data/nyu2_train/study_room_0005b_out/89.jpg')
image = cv2.imread('/home/lzq/DenseDepth-master/examples/499_image.png')
print(type(image))
print(image.shape) 
#自定义卷积核
kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
kernel_sharpen_2 = np.array([
        [1,1,1],
        [1,-7,1],
        [1,1,1]])
kernel_sharpen_3 = np.array([
        [-1,-1,-1,-1,-1],
        [-1,2,2,2,-1],
        [-1,2,8,2,-1],
        [-1,2,2,2,-1], 
        [-1,-1,-1,-1,-1]])/8.0
#图像锐化
kernel_sharpen_4 = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]])
kernel_sharpen_4_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
#图像模糊
kernel_sharpen_5 = np.array([
        [0.0625,0.125,0.0625],
        [0.125,0.25,0.125],
        [0.0625,0.125,0.125]])
#索贝尔
kernel_sharpen_6 = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]])
#浮雕
kernel_sharpen_7 = np.array([
        [-2,-1,0],
        [-1,1,1],
        [0,1,2]])
#大纲outline
kernel_sharpen_8 = np.array([
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]])
#拉普拉斯算子
kernel_sharpen_9 = np.array([
        [0,1,0],
        [1,-4,1],
        [0,1,0]])
#卷积
# output_1 = cv2.filter2D(image,-1,kernel_sharpen_1)
# output_2 = cv2.filter2D(image,-1,kernel_sharpen_2)
# output_3 = cv2.filter2D(image,-1,kernel_sharpen_3)
output_4 = cv2.filter2D(image,-1,kernel_sharpen_4)
output_4_1 = cv2.filter2D(image,-1,kernel_sharpen_4_1)
#output_5 = cv2.filter2D(image,-1,kernel_sharpen_5)
#dst = cv2.GaussianBlur(image, (11,11), 0) #高斯模糊
dst = cv2.bilateralFilter(image,3,140,140)
# output_6 = cv2.filter2D(image,-1,kernel_sharpen_6)
# output_7 = cv2.filter2D(image,-1,kernel_sharpen_7)
# output_8 = cv2.filter2D(image,-1,kernel_sharpen_8)
# output_9 = cv2.filter2D(image,-1,kernel_sharpen_9)
output_10 = cv2.filter2D(dst,-1,kernel_sharpen_4)
output_10_1 = cv2.filter2D(dst,-1,kernel_sharpen_4_1)
#显示锐化效果
cv2.imshow('Original Image',image)
# cv2.imshow('sharpen_1 Image',output_1)
# cv2.imshow('sharpen_2 Image',output_2)
# cv2.imshow('sharpen_3 Image',output_3)
cv2.imshow('H1_sharpen Image',output_4)
cv2.imshow('H2_sharpen Image',output_4_1)
#cv2.imshow('sharpen_5 Image',output_5)
cv2.imshow('vague',dst)
# cv2.imshow('sharpen_6 Image',output_6)
# cv2.imshow('sharpen_7 Image',output_7)
# cv2.imshow('sharpen_8 Image',output_8)
# cv2.imshow('sharpen_9 Image',output_9)
cv2.imshow('vague sharp H1',output_10)
cv2.imshow('vague sharp H2',output_10_1)
#停顿
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()

