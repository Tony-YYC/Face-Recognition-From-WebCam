# -*- coding: utf-8 -*-
import cv2
import os
import config
import torch
import time
from model import Net
from PIL import Image
from PIL import ImageFont,ImageDraw
import data_set
import numpy as np
# 检查是否有GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class catch_face:
    def __init__(self,tag):
        self.tag = tag
    def catch_video(self,window_name='catch face', camera_idx=0):
        cv2.namedWindow(window_name)
    # 视频来源，可以来自一段已存好的视频，也可以直接来自摄像头
        cap = cv2.VideoCapture(camera_idx,cv2.CAP_DSHOW)
        while cap.isOpened():
        # 读取一帧数据
            ok, self.frame = cap.read()
            if not ok:
                break
        # 抓取人脸的方法, 后面介绍
            self.catchface()
            # 输入'q'退出程序
            cv2.imshow(window_name, self.frame)
            c = cv2.waitKey(1)
            if c & 0xFF == ord('q'):
                break
            time.sleep(0.2)
    # 释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()

    def catchface(self):
        # 告诉OpenCV使用人脸识别分类器
        classfier = cv2.CascadeClassifier("/Users/yuyic/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
        # 识别出人脸后要画的边框的颜色，RGB格式
        color = (0, 255, 0)
        # 将当前帧转换成灰度图像
        grey = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)    
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        num = 1
        if len(face_rects) > 0: # 大于0则检测到人脸
            # 图片帧中有多个图片，框出每一个人脸
            for face_rects in face_rects:
                x, y, w, h = face_rects
                self.image = self.frame[y - 10:y + h + 10, x - 10:x + w + 10]
                # 保存人脸图像
                self.save_face(num)
                cv2.rectangle(self.frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                num += 1

    def save_face(self,num):
    # DATA_TRAIN为抓取的人脸存放目录，如果目录不存在则创建
        os.makedirs(os.path.join(config.DATA_TRAIN, str(self.tag)),exist_ok=True)
        img_name = os.path.join(config.DATA_TRAIN, str(self.tag), '{}_{}.jpg'.format(int(time.time()), num))
        # 保存人脸图像到指定的位置, 其中会创建一个tag对应的目录，用于后面的分类训练
        #cv2.imwrite(img_name, self.image)
        cv2.imencode('.jpg', self.image)[1].tofile(img_name)

class recognize_face:
    def __init__(self):
        fo = open(config.NAME_TXT,"r",encoding="utf-8")
        self.FACE_LABEL = fo.readlines()
    def recognize_video(self,window_name='face recognize', camera_idx=0):
        cv2.namedWindow(window_name)
        cap = cv2.VideoCapture(camera_idx,cv2.CAP_DSHOW)
        while cap.isOpened():
            ok, self.frame = cap.read()
            if not ok:
                break
            self.catchfacein()
            cv2.imshow(window_name, self.frame)
            c = cv2.waitKey(1)
            if c & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def catchfacein(self):
        classfier = cv2.CascadeClassifier("/Users/yuyic/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
        color = (0, 255, 0)
        grey = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(face_rects) > 0:
            for face_rects in face_rects:
                x, y, w, h = face_rects
                image = self.frame[y - 10:y + h + 10, x - 10:x + w + 10]
                # opencv 2 PIL格式图片
                PIL_image = self.cv2pil(image)
                # 使用模型进行人脸识别
                label = self.predict_model(PIL_image)
                cv2.rectangle(self.frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                # 将人脸对应人名写到图片上, 以为是中文名所以需要加载中文字体库
                self.frame = self.paint_chinese_opencv(self.frame,self.FACE_LABEL[label], (x-10, y+h+10), color)

    def cv2pil(self,image):
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def predict_model(self,image):
        data_transform = data_set.get_transform()
        # 对图片进行预处理，同训练的时候一样
        image = data_transform(image)
        image = image.view(-1, 3, 32, 32)
        net = Net().to(DEVICE)
        # 加载模型参数权重
        net.load_state_dict(torch.load(os.path.join(config.DATA_MODEL, config.DEFAULT_MODEL)))
        output = net(image.to(DEVICE))
        # 获取最大概率的下标
        pred = output.max(1, keepdim=True)[1]
        return pred.item()

    def paint_chinese_opencv(self,im, chinese, pos, color):
        img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # 引用字体库
        font = ImageFont.truetype(config.FONTS, 20)
        fillColor = color
        position = pos
        if not isinstance(chinese, str):
            chinese = str(chinese)
        draw = ImageDraw.Draw(img_PIL)
        # 写上人脸对应的人名
        draw.text(position, chinese, font=font, fill=fillColor)
        img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        return img