# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 14:35:33 2018
@author: Miracle
"""
 
import cv2
import numpy as np
#加载图像
image = cv2.imread('/home/lzq/nyu_data/train/nyu_data/data/nyu2_test/00013_colors.png')
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
output_1 = cv2.filter2D(image,-1,kernel_sharpen_1)
output_2 = cv2.filter2D(image,-1,kernel_sharpen_2)
output_3 = cv2.filter2D(image,-1,kernel_sharpen_3)
output_4 = cv2.filter2D(image,-1,kernel_sharpen_4)
output_5 = cv2.filter2D(image,-1,kernel_sharpen_5)
output_6 = cv2.filter2D(image,-1,kernel_sharpen_6)
output_7 = cv2.filter2D(image,-1,kernel_sharpen_7)
output_8 = cv2.filter2D(image,-1,kernel_sharpen_8)
output_9 = cv2.filter2D(image,-1,kernel_sharpen_9)
output_10 = cv2.filter2D(output_5,-1,kernel_sharpen_1)#先模糊后锐化
#显示锐化效果
cv2.imshow('Original Image',image)
cv2.imshow('sharpen_1 Image',output_1)
cv2.imshow('sharpen_2 Image',output_2)
cv2.imshow('sharpen_3 Image',output_3)
cv2.imshow('sharpen_4 Image',output_4)
cv2.imshow('sharpen_5 Image',output_5)
cv2.imshow('sharpen_6 Image',output_6)
cv2.imshow('sharpen_7 Image',output_7)
cv2.imshow('sharpen_8 Image',output_8)
cv2.imshow('sharpen_9 Image',output_9)
cv2.imshow('sharpen_10 Image',output_10)
#停顿
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()

