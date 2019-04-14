#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
 @project:face_encryption
 @file:face_dlib
 @author:HongDong_Zhao
 @create_time:2019/4/8 10:08 PM
"""
import dlib
import cv2
from skimage import io
from skimage.draw import polygon_perimeter

filename = '4.jpg'

detector = dlib.get_frontal_face_detector()
image = io.imread(filename)
#这里的参数可以控制识别人脸的精度
faces = detector(image, 2)

with open('test_dlib.txt', 'w') as f:
    for d in faces:
        rr, cc = polygon_perimeter([d.top(), d.top(), d.bottom(), d.bottom()], [d.right(), d.left(), d.left(), d.right()])
        image[rr, cc] = (0, 255, 0)
        f.write('{:5d} {:5d} {:5d} {:5d}\n'.format(d.left(),d.top(), (d.right()-d.left()),(d.bottom()-d.top())))

io.imsave('te2.jpg', image)


img = cv2.imread(filename)
sp =img.shape
sz1 = sp[0]
sz2 = sp[1]
sz3 = sp[2]
print('%d daxiao' %sp[0])
print('%d daxiao' % sp[1])
print('%d daxiao' % sp[2])

#for i in faces:
    #roi[i]= image[i.top():i.bottom(), i.left():i.right()]

#cv2.imshow('face',roi)
