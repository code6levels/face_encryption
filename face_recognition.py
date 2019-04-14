#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
 @project:face_encryption
 @file:face_recognition
 @author:HongDong_Zhao
 @create_time:2019/4/8 6:10 PM
"""
import cv2
import numpy as np

import os


filename = "11.jpg"


def detect(filename):
    # haarcascade_frontalface_default.xml存储在package安装的位置
    face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #传递参数是scaleFactor和minNeighbors,分别表示人脸检测过程中每次迭代时图像的压缩率以及每个人脸矩形保留近邻数目的最小值
    #检测结果返回人脸矩形数组 3的话识别准确率还可以
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    # 把四个坐标变量写入txt文件中
    with open('test.txt', 'w') as f:  # 打开test.txt   如果文件不存在，创建该文件
        for (face_x, face_y, face_w, face_h) in faces:
            #img 是原图；（x,y)是左上角的坐标值；（x+w，y+h）是矩阵的右下点坐标；（0,255,0）是画线对应的rgb颜色；2是所画的线的宽度
            img = cv2.rectangle(img, (face_x, face_y), (face_x + face_w, face_y + face_h), (0,255, 0), 1)

            f.write('{:5d} {:5d} {:5d} {:5d}\n'.format(face_x, face_y, face_w, face_h))  # 把变量var写入test.txt。这里var必须是str格式，如果不是，则可以转一下。
    cv2.namedWindow("Human Face Result!")
    cv2.imshow("Human Face Result!", img)
    cv2.imwrite("Face.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def jiami(a):
    leng=len(bin(a).replace('0b',''))
    if leng==8:   #如果位数为8
        return int('0b'+(bin(a).replace('0b','')[::-1]),2)
    else:     #之所以用else ,因为要先变成8位二进制
        return int('0b'+((8-leng)*'0'+bin(a).replace('0b',''))[::-1],2)



if __name__ == '__main__':

    img = cv2.imread(filename)
    sp =img.shape
    sz1 = sp[0]
    sz2 = sp[1]
    sz3 = sp[2]
    print('%d daxiao' %sp[0])
    print('%d daxiao' % sp[1])
    print('%d daxiao' % sp[2])
    detect(filename)

    # 读取test.txt文件中的多个坐标
    f = open('test.txt','r')
    a = f.readline()
    x = a.split()
    x = [int(x) for x in x if x ]
    print(x)
    bbox = np.stack(x).astype(np.float32)
    # 生成随机图片
    keyimg = np.zeros((sz1,sz2),img.dtype)
    cv2.imshow("keyimg",keyimg)

    #将随机生成的图片和filename文件进行相互异或



    cv2.waitKey(0)
    cv2.destroyAllWindows()


