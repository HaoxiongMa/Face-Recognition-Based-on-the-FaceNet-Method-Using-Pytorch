# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:30:30 2019

@author: Haoxiong Ma
"""

import tensorflow as tf
import align.detect_face as detect_face
import numpy as np


#using the MTCNN pretrained model to cut the face out of the image to 160x160
#files 'align' is from https://github.com/davidsandberg/facenet/tree/master/src/align

class Facedetection:
    def __init__(self):
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)
    
    def detect_face(self,image):
        bounding_boxes, points = detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        return  bounding_boxes, points

    def detect_face_pred(self,image,fixed=None):

        bboxes, landmarks = detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        landmarks_list = []
        landmarks=np.transpose(landmarks)
        bboxes=bboxes.astype(int)
        bboxes = [b[:4] for b in bboxes]
        for landmark in landmarks:
            face_landmarks = [[landmark[j], landmark[j + 5]] for j in range(5)]
            landmarks_list.append(face_landmarks)
        if fixed is not None:
            bboxes,landmarks_list=self.get_square_bboxes(bboxes, landmarks_list, fixed)
        return bboxes,landmarks_list

    def get_square_bboxes(self, bboxes, landmarks, fixed="height"):
        '''
        获得等宽或者等高的bboxes
        :param bboxes:
        :param landmarks:
        :param fixed: width or height
        :return:
        '''
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            center_x, center_y = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if fixed == "height":
                dd = h / 2
            elif fixed == 'width':
                dd = w / 2
            x11 = int(center_x - dd)
            y11 = int(center_y - dd)
            x22 = int(center_x + dd)
            y22 = int(center_y + dd)
            new_bbox = (x11, y11, x22, y22)
            new_bboxes.append(new_bbox)
        return new_bboxes, landmarks