#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.nets 
slim = tf.contrib.slim
import urllib

from preprocessing import vgg_preprocessing, inception_preprocessing

checkpoints_dir = './tmp/checkpoints/'
image_size = 224


def main():
    with tf.Graph().as_default():
        url = 'https://timgsa.baidu.com/timg?image&quality=80&size=b10000_10000&sec=1521798628&di=94d246dc237b56a2b1c218c14468576a&src=http://img.tradekey.com/p-5742368-20111223063617/kiwi-fruit.jpg'
        image_string = urllib.request.urlopen(url).read()
        image = tf.image.decode_jpeg(image_string, channels=3)
        processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images  = tf.expand_dims(processed_image, 0)
        
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            _, end_points = resnet_v1.resnet_v1_152(processed_images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'resnet_v1_152.ckpt'),
            slim.get_model_variables('resnet_v1_152'))
        
        with tf.Session() as sess:
            init_fn(sess)
            np_image, probabilities = sess.run([image, probabilities])
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
            


if __name__=='__main__':
    pass