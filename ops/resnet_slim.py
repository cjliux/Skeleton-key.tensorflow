import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slimnets


IMAGENET_MEAN_RGB = [123.68, 116.779, 103.939]


class ResNet(object):

    def __init__(self, images, cpkt_file):
        self.images = images
        data = self.images - np.array(IMAGENET_MEAN_RGB)
        with slim.arg_scope(slimnets.resnet_v1.resnet_arg_scope()):
            _, self.end_points = slimnets.resnet_v1.resnet_v1_152(inputs = data,
                                                        is_training = False,
                                                        global_pool = False,
                                                        reuse = tf.AUTO_REUSE)
        self.scope = "resnet_v1_152"
        self.resnet_variables = slim.get_model_variables(self.scope)
        self.features = self.end_points['resnet_v1_152/block4']

        self.init_fn = slim.assign_from_checkpoint_fn(
                cpkt_file, self.resnet_variables)

        



