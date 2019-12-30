# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:54:51 2019

@author: littlestrong
"""

import cv2
import numpy as np
from skimage import data_dir, io, transform, color
import os

# 车牌面积范围
MIN_AREA = 3000
MAX_AREA = 5000
# 车牌宽高比值
MIN_RATE = 1.0
MAX_RATE = 3.0
# 白色范围
sensitivity = 115
lower_white = np.array([0, 0, 255 - sensitivity])
upper_white = np.array([255, sensitivity, 255])


def preProcess(src):
    # 高斯平滑，中值滤波
    gaussian = cv2.GaussianBlur(src, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    median = cv2.medianBlur(gaussian, 5)
    # 将rgb模型转化为hsv模型，方便颜色定位
    # 根据阈值找到对应颜色
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    # mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # blue = cv2.bitwise_and(hsv, hsv, mask=mask_blue)

    #
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    white = cv2.bitwise_and(hsv, hsv, mask=mask_white)
    gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)

    # 灰度化
    # gray = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    # 形态学操作：膨胀 与 腐蚀
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilation = cv2.dilate(gray, element, iterations=1)
    erosion = cv2.erode(dilation, element, iterations=1)
    return erosion

def detect_edge(imageArr):  # 边缘进行分离
    img_copy = imageArr.copy()
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    kernel = np.ones((23, 23), np.uint8)
    img_opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(gray_img, 1, img_opening, -1, 0)
    # 找到图像边缘
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    # # 使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((10, 10), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    # # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for tmp_counter in contours:
        x, y, w, h = cv2.boundingRect(tmp_counter)
        cv2.rectangle(img_edge2, (x, y), (x + w, y + h), (255, 0, 0), 5)
    # cv2.imshow('3', img_edge2)
    return gray_img, contours

def combine_hsv_edge(img,gray, edge_contours, hsv_counters, Min_Area=700):
    """
    :param gray: gray_img
    :param edge_contours: edge_counter
    :param hsv_contours: hsv_counter
    :param Min_Area: min_area
    :return:
    """

    temp_contours = []
    for edge_contour in edge_contours:
        if cv2.contourArea(edge_contour) > Min_Area:
            for hsv_counter in hsv_counters:
                if cv2.contourArea(hsv_counter) < Min_Area:
                    continue
                box1 = cv2.boundingRect(edge_contour)
                box2 = cv2.boundingRect(hsv_counter)
                if overlap(box1, box2) > 100:
                    temp_contours.append(edge_contour)
    car_plate = []
    for temp_contour in temp_contours:
        rect_tupple = cv2.minAreaRect(temp_contour)
        rect_width, rect_height = rect_tupple[1]
        # if rect_width < rect_height:
        #         #     rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height
        # 车牌正常情况下宽高比在2 - 5.5之间
        if aspect_ratio > 0.5 and aspect_ratio < 1:
            x, y, w, h = cv2.boundingRect(temp_contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
            car_plate.append(temp_contour)
            rect_vertices = cv2.boxPoints(rect_tupple)
            rect_vertices = np.int0(rect_vertices)
    # cv2.imshow('4', gray)
    return car_plate

def overlap(box1, box2):
    """

    :param box1: [x,y,w,h]
    :param box2: [x,y,w,h]
    :return:
    """
    if box1[0] > box2[0] + box2[2]:
        return 0.0
    if box1[1] > box2[1] + box2[3]:
        return 0.0
    if box1[0] + box1[2] < box2[0]:
        return 0.0
    if box1[1] + box1[3] < box2[1]:
        return 0.0
    colInt = min(box1[0] + box1[2], box2[0] + box2[2]) - max(box1[0], box2[0])
    rowInt = min(box1[1] + box1[3], box2[1] + box2[3]) - max(box1[1], box2[1])
    intersection = colInt * rowInt
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    return intersection / area1 + area2 -intersection

# 过滤掉一些不符合要求地轮廓，返回轮廓的最小包围矩形
def getRect(contours):
    result = []
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        # 求出轮廓的最小包围矩形
        rect = cv2.minAreaRect(contour)
        # 长宽
        width, height = rect[1]
        if height == 0 or width == 0:
            continue
        rate = max(width, height) / min(width, height)
        # 过滤掉长宽比例不符合要求的轮廓
        if rate < MIN_RATE or rate > MAX_RATE:
            continue
        # 符合要求则添加进result
        result.append(rect)
    return result


# 根据矩形在原图裁剪图片
# src 原图  rect 最小包围矩形 padding 缩小矩形范围
def cutSrcByRect(src, rect, padding=0):
    box = cv2.boxPoints(rect)
    min_x = int(min(box[:, 0])) + padding
    max_x = int(max(box[:, 0])) - padding
    min_y = int(min(box[:, 1])) + padding
    max_y = int(max(box[:, 1])) - padding
    target = src[min_y:max_y, min_x:max_x]
    cv2.rectangle(src, (min_x, min_y), (max_x, max_y), (255, 0, 0), 5)
    return target


def location(src):
    imageorg = cv2.imread(src)
    print(imageorg.shape)
    imageorg = cv2.resize(imageorg, (1024, 512))
    image = preProcess(imageorg)
    # 将处理完的图像转变为三通道
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = getRect(contours)
    if len(result) != 0:
        dst = cutSrcByRect(imageorg, result[0])
        return imageorg
    else:
        dst = np.zeros([400, 400, 3], np.uint8)


# def file_name(file_dir):
#     L = []
#     for root, dirs, files in os.walk(file_dir):
#         for file in files:
#             if os.path.splitext(file)[1] == '.jpg':
#                 L.append(os.path.join(file))
#     return L


# L = file_name('/Users/luchixiang/Downloads/plate')
# ImagePath = '/Users/luchixiang/Downloads/output3'
# # 保存路径
# str = data_dir + ImagePath + '/*.jpg'
# coll = io.ImageCollection('D:\\plate\\*.jpg', load_func=location)
# for i in range(len(coll)):
#     io.imsave(ImagePath + '/' + L[i].split(".")[0] + '.jpg', coll[i])  # 循环保存图片
import glob
import os
os.makedirs('output_tradition',exist_ok=True)
for img_path in glob.glob('plate/*.jpg'):
    file_names = img_path.split('/')[-1]
    print(img_path)
    # src = cv2.imread(img_path)
    # img = cv2.resize(src, (1024, 512))
    # gray, edge_counter = detect_edge(img)
    # _, hsv_counter = detect_hsv(img)
    # combine_hsv_edge(img, gray, edge_counter, hsv_counter)
    # print('/Users/luchixiang/Downloads/output2/' + file_name)
    # cv2.imwrite('/Users/luchixiang/Downloads/output2/' + file_name, img)
    dst = location(img_path)
    cv2.imwrite('output_tradition/' + file_names, dst)