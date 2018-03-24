# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division

import heapq

import numpy as np
import utils
from utils import *

class Caption(object):
    def __init__(self, sentence, c, h, logprob, score, 
            embeds=None, contexts=None, hiddens=None, info=True):
        """
            Initializes the Caption.
            A data structure used to hold the information of a caption in the beam.
        Args:
          sentence: List of word ids in the caption.
          logprob: Log-probability of the caption.
          score: Score of the caption.
            None, a list of strings with the same length as 'sentence'.
        """
        self.sentence = sentence
        self.c = c
        self.h = h
        self.logprob = logprob
        self.score = score

        # extra information to be saved during decoding. (for later use)
        # e.g. intermediate status. 
        if info:
            self.embeds = embeds        # embedding of current stem word
            self.contexts = contexts    # context vector computed by level1.context4next
            self.hiddens = hiddens      # hidden state produced by level1 lstm 

    def __cmp__(self, other):
        """Compares Captions by score."""
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.
        The only method that can be called immediately after extract() is reset().
        Args:
            :sort: Whether to return the elements in descending sorted order.
        Returns:
            A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Reset the TopN to an empty state."""
        self._data = []


class CaptionGenerator(object):

    def __init__(self, model,
                 beam_size_1level=3, beam_size_2level=3,
                 max_caption_length_1level=16, max_caption_length_2level=6,
                 length_normalization_factor=0.0,
                 encourage_1level=0.0, encourage_2level=0.0,
                 level2=True):
        self.model = model
        self.vocab_1level = model.level1_word2ix
        self.vocab_2level = model.level2_word2ix
        
        self.beam_size_1level = beam_size_1level
        self.beam_size_2level = beam_size_2level
        self.max_caption_length_1level = max_caption_length_1level
        self.max_caption_length_2level = max_caption_length_2level
        self.length_normalization_factor = length_normalization_factor
        self.encourage_1level = encourage_1level
        self.encourage_2level = encourage_2level
        self.level2 = level2

    def beam_search(self, sess, img):
        """
        Params:
            :sess: tf session
            :img: image of shape (1, width, height, channels)
        Returns:
            top-ranked decoded literal sentence when level-2 is enabled(decode)
            otherwise, top-ranked digital sentence is returned.(no decode)
        """
        resnet = self.model.resnet
        level1 = self.model.level1_model

        # feed image into resnet and get image features
        image_features = sess.run(resnet.features, feed_dict={resnet.images: img})
        
        # level1 (skeleton)
        # initialize for beam search.
        (init_c, init_h, features_encode, features_proj) = sess.run(
                [level1.init_c, level1.init_h, 
                level1.features_encode, level1.features_proj],
                    feed_dict = {level1.image_features: image_features})
        
        initial_beam = Caption(
            sentence=[self.vocab_1level['START']],
            c=init_c, h=init_h,
            logprob=0.0, score=0.0, 
            embeds=[], contexts=[], hiddens=[])
        partial_captions = TopN(self.beam_size_1level)
        partial_captions.push(initial_beam)
        complete_captions = TopN(self.beam_size_1level)

        # Run beam search.
        for t in range(self.max_caption_length_1level):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()

            input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
            h_feed = np.reshape(np.array([c.h for c in partial_captions_list]), (-1, level1.dim_hid))
            c_feed = np.reshape(np.array([c.c for c in partial_captions_list]), (-1, level1.dim_hid))
            (c, h, log_softmax, alpha, context) = sess.run([level1.c, level1.h, 
                                level1.log_softmax, level1.alpha, level1.context4next],
                                    feed_dict={level1.c_feed: c_feed, 
                                               level1.h_feed: h_feed, 
                                               level1.in_word: input_feed, 
                                               level1.image_features: image_features})

            for i, partial_caption in enumerate(partial_captions_list):
                word_probabilities = log_softmax[i]
                word_probabilities[2:] += self.encourage_1level
                # For this partial caption, get the beam_size most probable next words.
                words_and_probs = list(enumerate(word_probabilities))
                words_and_probs.pop(level1._start)      # exclude START
                words_and_probs.sort(key=lambda x: -x[1])
                words_and_probs = words_and_probs[:self.beam_size_1level]

                # Each next word gives a new partial caption.
                for w, logp in words_and_probs:
                    if self.level2:
                        embed = sess.run(level1.embed4next, feed_dict={level1.word_feed: np.array([w])})
                    else:
                        embed = None
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + logp
                    score = logprob
                    if w == level1.word_to_idx['EOS']:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence) ** self.length_normalization_factor
                        beam = Caption(sentence, c[i], h[i], logprob, score,
                                       partial_caption.embeds, partial_caption.contexts, partial_caption.hiddens)
                        complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, c[i], h[i], logprob, score,
                                       partial_caption.embeds + [embed],
                                       partial_caption.contexts + [context[i]],
                                       partial_caption.hiddens + [h[i]])
                        partial_captions.push(beam)
                        
            if partial_captions.size() == 0:
                # We have run out of partial candidates; happens when beam_size = 1.
                break
        if not complete_captions.size():
            complete_captions = partial_captions

        level1_top_captions = complete_captions.extract(sort=True)

        full_sentence = []
        # level2 can be excluded for analysis
        if self.level2:
            level2 = self.model.level2_model
            # level2 (attributes)
            for caption in level1_top_captions:
                # for each caption(only one sentence)
                sentence_level1 = caption.sentence
                embeds, contexts, hiddens = caption.embeds, caption.contexts, caption.hiddens

                # only take the best skeleton generated from level1, 
                # and splitted as word sequence (be careful!!!)
                sent_level1 = utils.decode_captions(np.squeeze(np.asarray(sentence_level1)),
                                                    level1.idx_to_word)[0]
                words_level1 = sent_level1.split(' ') 

                attrs_level2 = []
                # iterate over the whole sentence word by word
                for t_level1 in range(len(embeds)):
                    # initialize for beam search.
                    embed = np.reshape(embeds[t_level1], (1, -1))
                    context = np.reshape(contexts[t_level1], (1, -1))
                    hidden = np.reshape(hiddens[t_level1], (1, -1))
                    (init_c, init_h) = sess.run([level2.init_c, level2.init_h],
                                                feed_dict={level2.embedding: embed, 
                                                           level2.context: context, 
                                                           level2.hidden: hidden})

                    initial_beam = Caption(
                                sentence=[self.vocab_2level['START']],
                                c=init_c, h=init_h,
                                logprob=0.0, score=0.0, info=False)
                    partial_captions = TopN(self.beam_size_2level)
                    partial_captions.push(initial_beam)
                    complete_captions = TopN(self.beam_size_2level)

                    # Run beam search.
                    for t in range(self.max_caption_length_2level):
                        partial_captions_list = partial_captions.extract()
                        partial_captions.reset()

                        input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
                        h_feed = np.reshape(np.array([c.h for c in partial_captions_list]), (-1, level2.dim_hid))
                        c_feed = np.reshape(np.array([c.c for c in partial_captions_list]), (-1, level2.dim_hid))
                        (c, h, log_softmax) = sess.run([level2.c, level2.h, level2.log_softmax],
                                                       feed_dict={level2.c_feed: c_feed, 
                                                                  level2.h_feed: h_feed,
                                                                  level2.in_word: input_feed})

                        for i, partial_caption in enumerate(partial_captions_list):
                            word_probabilities = log_softmax[i]
                            word_probabilities[2:] += self.encourage_2level
                            words_and_probs = list(enumerate(word_probabilities))
                            words_and_probs.pop(level2._start)  # exclude START
                            words_and_probs.sort(key=lambda x: -x[1])
                            words_and_probs = words_and_probs[0:self.beam_size_2level]

                            for w, logp in words_and_probs:
                                sentence = partial_caption.sentence + [w]
                                logprob = partial_caption.logprob + logp
                                score = logprob

                                if w == level2.word_to_idx['EOS']:
                                    if self.length_normalization_factor > 0:
                                        score /= len(sentence) ** self.length_normalization_factor
                                    beam = Caption(sentence, c[i], h[i], logprob, score, info=False)
                                    complete_captions.push(beam)
                                else:
                                    beam = Caption(sentence, c[i], h[i], logprob, score, info=False)
                                    partial_captions.push(beam)

                        if partial_captions.size() == 0:
                            break
                    if not complete_captions.size():
                        complete_captions = partial_captions

                    # exclude START, only top-ranked attr is used.
                    # attr ~ list([str <x1>])
                    attr = utils.decode_captions(
                        np.squeeze(np.asarray(
                                complete_captions.extract(sort=True)[0].sentence
                            ))[1:], 
                        level2.idx_to_word)
                    # append str to list
                    attrs_level2.extend(attr) 
                full_sentence.append(' '.join([i + ' ' + j if i != '' else j for (j, i) in zip(words_level1, attrs_level2)]))
        else:
            # exclude START
            full_sentence = [i.sentence[1:] for i in level1_top_captions]
            full_sentence = utils.decode_captions(np.asarray(full_sentence), level1.idx_to_word)
        
        # only return top-ranked stem with attr.
        return full_sentence[0]
    