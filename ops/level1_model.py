import os
import tensorflow as tf
import h5py
import pickle as pkl
import numpy as np
import ops.resnet as resnet

class Level1Model(object):
    def __init__(self, word_to_idx,
                 n_feats = 49, dim_in_feat=2048, dim_feat=512, conv_ksize=3, 
                 dim_embed=512, dim_factor=512,  dim_hidden=1800,
                 alpha_c=0.0, dropout=True):
        # vocab info
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.vocab_size = len(word_to_idx)
        assert self.vocab_size == max(self.idx_to_word.keys())+1

        self._start = word_to_idx['START']
        self._eos = word_to_idx['EOS']
        self._unk = word_to_idx['UNK']

        # hyper params
        self.alpha_c = alpha_c
        self.dropout = dropout
        self.n_in_feats = dim_in_feat
        self.n_feats = n_feats
        self.dim_in_feat = dim_in_feat
        self.dim_feat = dim_feat
        self.dim_embed = dim_embed
        self.dim_factor = dim_factor
        self.dim_hid = dim_hidden
        self.conv_ksize = conv_ksize

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        # feature after cnn_encoding.
        self.features_encode = None
        # feature after projection for attention.
        self.features_proj = None
        # initial states from image.
        self.init_c = None
        self.init_h = None
        # state and info from steps.
        self.c = None
        self.h = None
        self.log_softmax = None
        self.alpha = None
        self.context4next = None
        # embedding 4 next layer
        self.embed4next = None


    def _get_init_lstm(self, features):
        with tf.variable_scope('level1/init_lstm'):
            features_mean = tf.reduce_mean(features, 1)    

            w_1 = tf.get_variable('w1', shape=[self.dim_feat, self.dim_hid], 
                initializer=self.weight_initializer, dtype=tf.float32)
            b_1 = tf.get_variable('b1', shape=[self.dim_hid], 
                initializer=self.const_initializer, dtype=tf.float32)
            w_2 = tf.get_variable('w2', shape=[self.dim_hid, self.dim_hid*2], 
                initializer=self.weight_initializer, dtype=tf.float32)
            b_2 = tf.get_variable('b2', shape=[self.dim_hid*2], 
                initializer=self.const_initializer, dtype=tf.float32)

            h1 = tf.nn.relu(tf.matmul(features_mean, w_1) + b_1)

            init = tf.nn.tanh(tf.matmul(h1, w_2) + b_2)
            init = tf.reshape(init, shape=[-1, 2, self.dim_hid])
            init_c, init_h = tf.unstack(init, axis=1)
            return init_c, init_h

    def _cnn_encoding(self, features):
        with tf.variable_scope('level1/cnn_encoding'):
            weight = tf.get_variable('weight', 
                shape=[self.conv_ksize, self.conv_ksize, self.dim_in_feat, self.dim_feat], 
                initializer=self.weight_initializer, dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[self.dim_feat], 
                initializer=self.const_initializer, dtype=tf.float32)

            conv = tf.nn.conv2d(features, weight, [1, 1, 1, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, bias)
            conv_reshape = tf.reshape(conv, [-1, self.n_feats, self.dim_feat])  # (-1, 49, 512)
            return conv_reshape

    def _word_embedding(self, inputs, reuse=False):
        """ default to be initialized with self.w2v
        """
        # todo: word_embedding for <START> token is all-zero
        with tf.variable_scope('level1/word_embedding', reuse=reuse):
            w = tf.get_variable('w', shape=[self.vocab_size, self.dim_embed], 
                    initializer=self.emb_initializer,  dtype=tf.float32)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        """ff projection over features
        Inputs: features ~ (B, L, D)
        Ouputs: features_proj ~ (B, L, D)
        """
        with tf.variable_scope('level1/project_features'):
            # features_proj --> proj_ctx
            # todo: features_proj = tf.matmul(features_flat, w) + b
            w1 = tf.get_variable('weight1', shape=[self.dim_feat, self.dim_feat], 
                initializer=self.weight_initializer, dtype=tf.float32)
            b1 = tf.get_variable('bias1', shape=[self.dim_feat], 
                initializer=self.const_initializer, dtype=tf.float32)
            w2 = tf.get_variable('weight2', shape=[self.dim_feat, self.dim_feat], 
                initializer=self.weight_initializer, dtype=tf.float32)
            b2 = tf.get_variable('bias2', shape=[self.dim_feat], 
                initializer=self.const_initializer, dtype=tf.float32)
            
            features_flat = tf.reshape(features, [-1, self.dim_feat])
            features_proj1 = tf.nn.tanh(tf.matmul(features_flat, w1) + b1)
            features_proj = tf.matmul(features_proj1, w2) + b2
            features_proj = tf.reshape(features_proj, [-1, self.n_feats, self.dim_feat])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        """Compute Attention(h, features_proj)
        Note: 
            features is not used.
        """
        with tf.variable_scope('level1/attention_layer', reuse=reuse):
            w = tf.get_variable('w', shape=[self.dim_hid, self.dim_feat], 
                initializer=self.weight_initializer, dtype=tf.float32)
            b = tf.get_variable('b', shape=[self.dim_feat], 
                initializer=self.const_initializer, dtype=tf.float32)
            w_att = tf.get_variable('w_att', shape=[self.dim_feat, 1], 
                initializer=self.weight_initializer, dtype=tf.float32)
            b_att = tf.get_variable('b_att', shape=[1], 
                initializer=self.const_initializer, dtype=tf.float32)

            h_att = tf.nn.tanh(features_proj + tf.expand_dims(tf.matmul(h, w) + b, 1))
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.dim_feat]), w_att) + b_att, [-1, self.n_feats])
            alpha = tf.nn.softmax(out_att)

            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')
            return context, alpha

    def _lstm(self, input_h, input_c, input_x, context, reuse=False):
        """One step iteration of level1 lstm
        """
        with tf.variable_scope('level1/lstm', reuse=reuse):
            w_i2h = tf.get_variable('w_i2h', shape=[self.dim_embed, 4*self.dim_hid], 
                initializer=self.weight_initializer, dtype=tf.float32)
            w_h2h = tf.get_variable('w_h2h', shape=[self.dim_hid, 4*self.dim_hid], 
                initializer=self.weight_initializer, dtype=tf.float32)
            w_z2h = tf.get_variable('w_z2h', shape=[self.dim_feat, 4*self.dim_hid], 
                initializer=self.weight_initializer, dtype=tf.float32)
            b_all = tf.get_variable('b_z2h', shape=[4*self.dim_hid], 
                initializer=self.const_initializer, dtype=tf.float32)

            input_x = tf.cast(input_x, tf.float32)
            i2h = tf.matmul(input_x, w_i2h) 
            h2h = tf.matmul(input_h, w_h2h) 
            z2h = tf.matmul(context, w_z2h) 
            all_input_sums = i2h + h2h + z2h + b_all
            reshaped = tf.reshape(all_input_sums, [-1, 4, self.dim_hid])
            n1, n2, n3, n4 = tf.unstack(reshaped, axis=1)
            in_gate = tf.sigmoid(n1)
            forget_gate = tf.sigmoid(n2)
            out_gate = tf.sigmoid(n3)
            in_transform = tf.tanh(n4)
            c = tf.multiply(forget_gate, input_c) + tf.multiply(in_gate, in_transform)
            h = tf.multiply(out_gate, tf.tanh(c))
            return c, h

    def _selector(self, context, h, reuse=False):
        """ gating context with beta gate."""
        with tf.variable_scope('level1/selector', reuse=reuse):
            w = tf.get_variable('w', shape=[self.dim_hid, self.dim_feat], 
                initializer=self.weight_initializer, dtype=tf.float32)
            b = tf.get_variable('b', shape=[self.dim_feat], 
                initializer=self.const_initializer, dtype=tf.float32)

            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        # pre-activated logits in softmax prediction layer
        with tf.variable_scope('level1/logits', reuse=reuse):
            w_hid = tf.get_variable('w_hid', shape=[self.dim_hid, self.dim_factor], 
                initializer=self.weight_initializer, dtype=tf.float32)
            w_ctx = tf.get_variable('w_ctx', shape=[self.dim_feat, self.dim_factor], 
                initializer=self.weight_initializer, dtype=tf.float32)
            b_log = tf.get_variable('b_log', shape=[self.dim_factor], 
                initializer=self.const_initializer, dtype=tf.float32)

            w_out = tf.get_variable('w_out', shape=[self.dim_factor, self.vocab_size], 
                initializer=self.weight_initializer, dtype=tf.float32)
            b_out = tf.get_variable('b_out', shape=[self.vocab_size], 
                initializer=self.const_initializer, dtype=tf.float32)

            x = tf.cast(x, tf.float32)
            h_logits = x + tf.matmul(h, w_hid) + tf.matmul(context, w_ctx) + b_log 
            h_logits = tf.nn.tanh(h_logits)

            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits


    def build_inference(self, image_features):
        self.image_features = image_features
        self.c_feed = tf.placeholder(tf.float32, [None, self.dim_hid])
        self.h_feed = tf.placeholder(tf.float32, [None, self.dim_hid])
        self.in_word = tf.placeholder(tf.int32, [None])

        self.features_encode = self._cnn_encoding(features=self.image_features)
        self.init_c, self.init_h = self._get_init_lstm(features=self.features_encode)
        self.features_proj = self._project_features(features=self.features_encode)

        """one step inference of level1 (t>0)."""
        x = self._word_embedding(inputs=self.in_word, reuse=tf.AUTO_REUSE)
        # context computation
        context_pre, self.alpha = self._attention_layer(self.features_encode, 
                                                self.features_proj, self.h_feed, reuse=tf.AUTO_REUSE)
        context, beta = self._selector(context_pre, self.h_feed, reuse=tf.AUTO_REUSE)
        # lstm step with softmax prediction
        (self.c, self.h) = self._lstm(self.h_feed, self.c_feed, x, context, reuse=tf.AUTO_REUSE)
        self.log_softmax = tf.nn.log_softmax(self._decode_lstm(x, self.h, context, reuse=tf.AUTO_REUSE))

        # for level2 initialization
        self.context4next = tf.reduce_sum(
            tf.reshape(self.image_features, [-1, self.n_feats, tf.shape(self.image_features)[-1]]) 
            * tf.expand_dims(self.alpha, 2), 1)


    def build_info_for2layer(self):
        self.out_word = tf.placeholder(tf.int32, [None])
        # for level2 initialization
        self.embed4next = tf.reshape(self._word_embedding(inputs=self.out_word, reuse=tf.AUTO_REUSE), (-1,))

    # used for training
    def build_training(self, captions, mask, keep_prob):
        """Notes: caption should include START and EOS."""
        self.captions = captions
        self.mask = mask
        self.keep_prob = keep_prob

        caps = self.captions
        caps_in, caps_out = caps[:, :-1], caps[:, 1:]
        mask = self.mask[:,1:]
        n_samples, n_steps = tf.shape(caps_in)[0], tf.shape(caps_in)[1]
        emb_in = self._word_embedding(inputs=caps_in, reuse=tf.AUTO_REUSE)

        alpha_list = []
        
        def body(t, c_, h_, xent):
            context_pre, alpha = self._attention_layer(
                self.features_encode, self.features_proj, h_, reuse=tf.AUTO_REUSE)
            alpha_list.append(alpha) 
            context, beta = self._selector(context_pre, h_, reuse=tf.AUTO_REUSE)
            (c, h) = self._lstm(h_, c_, emb_in[:, t, :], context, reuse=tf.AUTO_REUSE)

            if self.dropout:
                h_drop = tf.nn.dropout(h, self.keep_prob)
            
            logits = self._decode_lstm(emb_in[:, t, :], h_drop, context, reuse=tf.AUTO_REUSE)
            # save intermediate result.
            # predict_list.append(tf.nn.top_k(logits)[1]) 

            xent += tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=caps_out[:, t]) * mask[:, t])

            return t+1, c, h, xent

        _, _, _, xent = tf.while_loop(
            cond=lambda t, c_, h_, xent: t < n_steps,
            body=body,
            loop_vars=[tf.constant(0), self.init_c, self.init_h, tf.constant(0.)],
            name='xent')

        self.loss = xent / tf.cast(n_samples, tf.float32)
        
        # regularization over attention weights:
        #   all words are equally covered in the generation process.
        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2)) 
            alphas_all = tf.reduce_sum(alphas, 1) #BD
            # todo: this is not the same with what we used in our model, but is better
            alpha_reg = self.alpha_c * tf.reduce_sum((1.*n_steps / self.n_feats - alphas_all) ** 2)
            self.loss += alpha_reg

        return self.loss
