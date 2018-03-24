import numpy as np
import pickle as pkl
# import hickle
import time
import os
import h5py
import json
import random


def load_coco_data(data_root, split='train', ret_idx=False):
    data_path = os.path.join(data_root, split)
    start_t = time.time()
    
    with open(os.path.join(data_path, split+'_captions.pkl'), 'rb') as fd:
        stems_list, stem_attrs_list = pkl.load(fd)
    f = h5py.File(os.path.join(data_path, split+'_images.h5'), 'r')
    image2stem = f['image2stem'][:]
    stem2image = f['stem2image'][:]
    images = f['images'][:]
    f.close()

    if ret_idx:
        with open(os.path.join(data_path, split+'_idx.pkl'), 'rb') as fd:
            idx = pkl.load(fd)

    end_t = time.time()
    print("Elapse time: %.2f" % (end_t - start_t))
    if ret_idx:
        return stems_list, stem_attrs_list, images, image2stem, stem2image, idx
    else:
        return stems_list, stem_attrs_list, images, image2stem, stem2image

def crop_image(data, training):
    """
        random crop if is training.
        otherwise, bounding box will be fixed at the center.
    """
    n_data = data.shape[0]
    # new_data = None
    if not training:
        # central crop
        start = int((256-224)/2)
        new_data = data[:, start:224+start, start:224+start, :]
    else:
        new_data = np.zeros((n_data, 224, 224, 3), dtype=np.float32)
        for i in range(n_data):
            start_x = random.randint(0, 256 - 224)
            start_y = random.randint(0, 256 - 224)
            new_data[i, :, :, :] = data[i, start_x:start_x+224, start_y:start_y+224, :]
    return new_data

def list2batch(stems, attrs=None):
    n_seqs = len(stems)
    maxlen = max(len(s) for s in stems)

    x = np.zeros((n_seqs, maxlen), np.int32)
    m = np.zeros((n_seqs, maxlen), np.float32)
    for i, s in enumerate(stems):
        x[i,:len(s)] = s
        m[i,:len(s)] = 1
    
    if attrs is None:
        return x, m
    else:
        maxlen_a = max(len(att) for attr in attrs for att in attr)
        x_a = np.zeros((n_seqs, maxlen, maxlen_a), np.int32)
        m_a = np.zeros((n_seqs, maxlen, maxlen_a), np.float32)
        
        for i, attr in enumerate(attrs):
            for j, att in enumerate(attr):
                x_a[i,j,:len(att)] = att
                m_a[i,j,:len(att)] = 1
        return x, m, x_a, m_a
        

def decode_helper(caption, idx_to_word):
    """
    Params:
        :captions: 1-dim array or list representing one sequence of word id
    Returns:
        a list of literal word.
    """
    words = []
    for t in range(len(caption)):
        word = idx_to_word[caption[t]]
        
        # skipping any special token
        if word == 'EOS': break
        if word == 'START': continue
        if word != 'NULL' and word != 'UNK':
            words.append(word)
    
    return words


def decode_captions(captions, idx_to_word):
    """
    Params:
        :captions: can either be 1-dim array representing one sequence of word id, 
            or 2-dim array representing a list of such sequences.
    Returns:
        a list of literal sentences.
    """
    words_all = []
    if captions.ndim == 2 or isinstance(captions[0], list):
        for cap in captions:
            words_all.append(decode_helper(cap, idx_to_word))
    else:
        words_all.append(decode_helper(captions, idx_to_word))
    return [' '.join(words) for words in words_all]


def decode_captions_2level(level1_cap, level2_cap, level1_idx2word, level2_idx2word):
    first_levels = decode_helper(level1_cap, level1_idx2word)
    decodes = []
    for first_level, second_level in zip(first_levels, level2_cap):
        attrs_decoded = decode_helper(second_level, level2_idx2word)
        decode_this = []
        for i, first_word in enumerate(first_level):
            decode_this.extend(attrs_decoded[i])
            decode_this.append(first_word)
        decodes.append(' '.join(decode_this))
    return decodes


def write_bleu(scores, path, epoch):
    if epoch == 0:
        file_mode = 'w'
    else:
        file_mode = 'a'
    with open(os.path.join(path, 'val.bleu.scores.txt'), file_mode) as f:
        f.write('Epoch %d\n' %(epoch+1))
        f.write('Bleu_1: %f\n' %scores['Bleu_1'])
        f.write('Bleu_2: %f\n' %scores['Bleu_2'])
        f.write('Bleu_3: %f\n' %scores['Bleu_3'])  
        f.write('Bleu_4: %f\n' %scores['Bleu_4']) 
        f.write('METEOR: %f\n' %scores['METEOR'])  
        f.write('ROUGE_L: %f\n' %scores['ROUGE_L'])  
        f.write('CIDEr: %f\n\n' %scores['CIDEr'])
