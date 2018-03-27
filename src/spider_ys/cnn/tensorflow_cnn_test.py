#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .tensorflow_cnn_train import (
    convert2gray, crack_captcha_cnn,
    vec2text, CHAR_SET_LEN,
    X, MAX_CAPTCHA, keep_prob, tf
)
import numpy as np
from PIL import Image
import os

# 定义CNN
output = crack_captcha_cnn()
saver = tf.train.Saver()
sess = tf.Session()
model_path = os.path.dirname(os.path.abspath(__file__)) + "/model"
saver.restore(sess, tf.train.latest_checkpoint(model_path))
predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)


def crack_captcha(image):
    image = np.array(image)
    image = convert2gray(image)  # 生成一张新图
    image = image.flatten() / 255  # 将图片一维化

    text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    return vec2text(vector)


if __name__ == '__main__':
    right_num = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    file_list = os.listdir("test_set")
    for f in file_list:
        image = Image.open("test_set/%s" % f).crop((0, 5, 160, 65))
        predict_text = crack_captcha(image)  # 导入模型识别
        real = f[0:4]
        right = 0
        for i in range(4):
            if predict_text[i] == real[i]:
                right += 1
        right_num[right] += 1
        print("正确：%s 预测: %s" % (real, predict_text))
    print("全部: %d 正确: %s" % (len(file_list), right_num))
