import os
import tensorflow as tf
import json
import ops.resnet_slim as resnet
import ops.level1_model as level1_model


ModeKeys = tf.contrib.learn.ModeKeys


class HierarchicalModel(object):

    def __init__(self, config, mode, build_level2=False):
        self.config = config
        self.mode = mode
        self.build_level2 = build_level2
        
        with open(os.path.join(self.config.data_root, 'train/word_to_ix_stem.json')) as fd:
            self.level1_word2ix = json.load(fd)
        with open(os.path.join(self.config.data_root, 'train/word_to_ix_attr.json')) as fd:
            self.level2_word2ix = json.load(fd)

    def build(self):
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'images')
        self.resnet = resnet.ResNet(self.images, cpkt_file=self.config.resnet_cpkt)

        self.level1_model = level1_model.Level1Model(word_to_idx=self.level1_word2ix,
                                                     n_feats=self.config.LEVEL1_n_feats,
                                                     dim_in_feat=self.config.LEVEL1_dim_in_feat,
                                                     dim_feat=self.config.LEVEL1_dim_feat,
                                                     dim_embed=self.config.LEVEL1_dim_embed,
                                                     dim_factor=self.config.LEVEL1_dim_factor,
                                                     dim_hidden=self.config.LEVEL1_dim_hidden,
                                                     conv_ksize=self.config.LEVEL1_conv_ksize,
                                                     alpha_c=self.config.LEVEL1_alpha, 
                                                     dropout=self.config.LEVEL1_dropout)
        self.image_features = tf.placeholder(tf.float32, [None, 7, 7, 2048], 'image_features')
        self.level1_model.build_inference(self.image_features)
        
        if self.build_level2:
            self.level1_model.build_info_for2layer()
            #self.level2_model = level2_model.Level2Model(word_to_idx=self.level2_word2ix,
            #                                             dim_feature=config.LEVEL2_dim_feature,
            #                                             dim_embed=config.LEVEL2_dim_embed,
            #                                             dim_hidden=config.LEVEL2_dim_hidden,
            #                                             dropout=config.LEVEL2_dropout, 
            #                                             n_time_step=config.LEVEL2_T)
            #self.level2_model.build_inference()
        
        if self.mode == ModeKeys.TRAIN:
            self.captions = tf.placeholder(tf.int32, [None, None])
            self.mask = tf.placeholder(tf.float32, [None, None])
            self.keep_prob = tf.placeholder(tf.float32, [])
            
            self.loss = self.level1_model.build_training(self.captions, self.mask, self.keep_prob)
            if self.build_level2:
                self.loss += self.level2_model.build_training()
            return self.loss
