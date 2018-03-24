import json
import time
import string
import h5py
import argparse
import nltk
from nltk.parse.stanford import StanfordParser
from nltk.tag.stanford import StanfordPOSTagger
import numpy as np
import scipy.misc as smisc
import os
import re
import pickle as pkl
from collections import defaultdict
from operator import itemgetter
"""
python 3.6.2
"""

#sent_parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
#st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
#data_root = './data/'


def prepro(sent):
    #temp = str(sent).lower()translate(None, string.punctuation).strip()
    # 'hello world ! hi , are you ok ?'
    temp = re.sub("[%s]"%re.escape(string.punctuation), '', str(sent).lower()).strip()
    # 'hello world  hi  are you ok'
    try:
        int(temp.split(' ')[-1])                # if end with int.
        temp = ' '.join(temp.split(' ')[:-1])   # then exclude it
    except:
        pass
    token = nltk.word_tokenize(temp)    # ['hello', 'world', 'hi', 'are', 'you', 'ok']
    return token, nltk.pos_tag(token)  
    # [('hello', 'JJ'), ('world', 'NN'), ('hi', 'NN'), ('are', 'VBP'), ('you', 'PRP'), ('ok', 'JJ') ]


def parse_dfs(sent_this, print_=False):
    visited = [(True, sent_this[0])]    # stack
    stem, loc = [], []
    sent_flatten = sent_this[0].leaves()
    loc_pointer = len(sent_flatten)
    while len(visited) != 0:
        # get top node from stack
        curr = visited[-1][1]
        curr_label = curr.label()
        record_this = visited[-1][0]
        visited.pop(-1)

        if curr.height() == 2:
            loc_pointer -= 1
            if record_this or loc_pointer == len(sent_flatten) - 1:
                stem.append(curr[0])
                loc.append(loc_pointer)
            continue
        
        for idx, i in enumerate(curr):
            record = True
            if i.height() <= 2:
                if curr_label == 'ADJP' and loc_pointer == len(sent_flatten):
                    #print 'HERE***', sent_flatten, curr
                    pass
                else:
                    if i.label()[0] in string.punctuation or (
                            curr_label in ('ADJP') and not (i[0] in ('next', 'full', 'ready'))
                            and not (idx == len(curr) - 1 and i.label() == 'NN')):
                        record = False
                    if curr_label == 'NP':
                        if not (
                                        (idx > 0 and curr[idx - 1].label().startswith('N') and i.label() == 'CC')
                                    or (i.label().startswith('N') and idx < len(curr) - 1 and curr[idx + 1].label() == 'CC')
                                or idx == len(curr) - 1
                            or (idx == len(curr) - 2 and curr[idx+1].label() == 'VBG')
                        ):
                            record=False
            visited.append((record, i))
    
    if print_:
        print(sent_this[0].pretty_print())
        print(stem[::-1], loc[::-1])
        print('++++++++++++++++++++++++++++++++++++++++++++')
    # split sentence into multiple stem-attr pairs
    stem_pairs = []
    prev = 0
    for stem_i, i in enumerate(loc[::-1]):
        this_ = {sent_flatten[i]:sent_flatten[prev:i]}
        stem_pairs.append(this_)
        prev = i + 1
    return stem, stem_pairs


def refine_stem(stem_attr):
    """
        deal with unusual cases that resulted in wrong extractions.
    Inputs:
        stem_attr: list({ stem_word : list(attr_word) }) 
    Returns:
        stem: list(stem_word)
        stem_attr: list({ stem_word : list(attr_word) }) 
    """
    new_stem_attr = []
    
    modify = False
    new_attr_to_add = None
    for temp in stem_attr:
        skeleton, attr = list(temp.items())[0]

        if new_attr_to_add is not None:
            attr = new_attr_to_add + attr
            new_attr_to_add = None

        if len(attr) > 0 and attr != ['a'] and skeleton in ('grazing', 'flying', 'standing',
                        'plays', 'sits', 'stands',
                        'on', 'of', 'with', 'below', 'in', 'by', 'between', 'along', 'near',
                        'from', 'behind', 'above', 'at', 'down', 'while', 'around',
                        'stand', 'play', 'sit'):
                new_sk = attr[-1]
                new_attr = attr[:-1]
                new_stem_attr.append({new_sk: new_attr})
                new_stem_attr.append({skeleton: []})
                modify = True
        elif skeleton == 'looking' and len(attr) > 0:
            if len(new_stem_attr) > 0 and list(new_stem_attr[-1].keys())[0] == 'very':
                new_attr_to_add = list(new_stem_attr[-1].values())[0] + ['very'] + attr + ['looking']
                new_stem_attr = new_stem_attr[:-1]
            else:
                new_attr_to_add = attr + ['looking']
            modify = True
        elif skeleton == 'colored':
            if len(new_stem_attr) > 0:
                new_attr_to_add = list(new_stem_attr[-1].values())[0] + [list(new_stem_attr[-1].keys())[0]] + attr + [skeleton]
                new_stem_attr = new_stem_attr[:-1]
            else:
                new_attr_to_add = attr + [skeleton]
        else:
            new_stem_attr.append({skeleton: attr})

    new_stem = [list(i.keys())[0] for i in new_stem_attr]
    if modify:
        print(stem_attr)
        print('-------->')
        print(new_stem_attr)
        print(new_stem)

    return new_stem, new_stem_attr


def parsing_coco():
    info_raw = json.load(open(os.path.join(data_root, 'coco_raw.json'), encoding='utf8'))
    caption_list = []
    for i in info_raw:
        caption_list.extend(i['captions'])
    print(len(caption_list))

    caption_stem_all = []
    caption_attr_all = []
    # iterate chunkwise over list of captions for stem extraction
    end_idx = 5000
    caption_chunk = caption_list[:end_idx]
    while len(caption_chunk) > 0:  
        t1 = time.time()

        token_pos = []
        for i in caption_chunk:
            token_pos.append(prepro(i))
        token, postag = zip(*token_pos)
        postag_stanford = st.tag_sents(token)
        # chose postag between nltk & stanford (for less N*)
        tags = []
        for i,j in zip(postag, postag_stanford):
            temp1 = [1 if ii[1].startswith('N') else 0 for ii in i]
            temp2 = [1 if ii[1].startswith('N') else 0 for ii in j]
            if sum(temp1) > sum(temp2):
                tags.append(j)
            else:
                tags.append(i)
        parse = sent_parser.tagged_parse_sents(tags)
        # extract stem
        stems, attrs = [], []
        for caption, sent in zip(caption_chunk, iter(parse)):
             # last_word = str(caption).translate(None, string.punctuation).strip().split(' ')[-1]
            # two steps stem extraction: parse -> refine (considered as black box)
            stem, stem_attr_pair = parse_dfs(list(sent), print_=False)
            # stem is also saved as keys in stem_attr_pair
            new_stem, new_stem_attr = refine_stem(stem_attr_pair)
            # save result: stem & (stem, attr)
            stems.append(new_stem)
            attrs.append(new_stem_attr)
        
        t2 = time.time()
        print(end_idx, 'processing time:', t2 - t1, stems[-1])
        # chunk summary
        caption_stem_all.extend(stems)
        caption_attr_all.extend(attrs)
        # next chunk 
        caption_chunk = caption_list[end_idx:end_idx+5000]
        end_idx += 5000
    # end chunk iteration

    # group result by caption instance and save to file.
    all_info = []
    for j, k in zip(caption_stem_all, caption_attr_all):
        all_info.append((j, k))
    with open(os.path.join(data_root,'caption_stem_attr_all.json'), 'w', encoding='utf8') as f:
        json.dump(all_info, f)
    # sanity check.
    assert(len(caption_stem_all) == len(caption_list))


def combine_result():
    info_raw = json.load(open(os.path.join(data_root, 'coco_raw.json'), encoding='utf8'))
    info_new = json.load(open(os.path.join(data_root, 'caption_stem_attr_all.json'), encoding='utf8'))
    
    info_all = []
    count = 0   # caption index in caption list(or caption-stem-attr list)
    for i in info_raw:
        attrs = []
        stems = []
        # combine stem-attr pairs from multiple instances
        for j in i['captions']:
            # temp = str(j).lower().translate(None, string.punctuation).strip().split()
            stem_this, attr_this = info_new[count]
            
            stems.append(stem_this)
            attrs.append(attr_this)
            count += 1
        # save them to augment the original coco json
        i['attrs'] = attrs
        i['stems'] = stems
        info_all.append(i)
    # save result 
    json.dump(info_all, open(os.path.join(data_root, 
        'coco_raw_with_attr.json'), 'w', encoding='utf8'))


###################################
### then this is the to-h5 part ###
###################################

def build_vocab_stem(imgs, params):
    """
    Returns:
        :vocab: a list of words (freq > word_count_threshold)
    """
    count_thr = params['stem_word_count_thr']

    # count up the number of words
    counts = {}
    for img in imgs:
        for txt in img['stems']:
            for w in txt:
                counts[w] = counts.get(w, 0) + 1
    total_words = sum(counts.values())
    wc = sorted(list(counts.items()), key=itemgetter(1), reverse=True)
    
    vocab = ['EOS', 'START']
    vocab.extend([w for w, n in wc if n >= count_thr])
    vocab.append('UNK')

    bad_words = [w for w, n in counts.items() if n < count_thr]
    
    # print some stats
    print('total words:', total_words)
    print('top words and their counts:')
    print('\n'.join(map(str, wc[:20])))
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    
    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for txt in img['stems']:
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    n_sents = sum(sent_lengths.values())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    for i in range(max_len + 1):
        n_len = sent_lengths.get(i, 0)
        print('%2d: %10d   %f%%' % (i, n_len, n_len * 100./n_sents))

    return vocab


def build_vocab_attr(imgs, params):
    count_thr = params['attr_word_count_thr']

    # count up the number of words
    counts = {}
    for i in imgs:
        for a in i['attrs']:
            for sa in a:
                for w in list(sa.values())[0]:
                    counts[w] = counts.get(w, 0) + 1
    total_words = sum(counts.values())
    wc = sorted(list(counts.items()), key=itemgetter(1), reverse=True)

    vocab = ['EOS', 'START']
    vocab.extend([w for w, n in wc if n >= count_thr])
    vocab.append('UNK')

    # print some stats
    bad_words = [w for w, n in counts.items() if n < count_thr]
    print('total words:', total_words)
    print('top words and their counts:')
    print('\n'.join(map(str, wc[:20])))
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    
    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for i in imgs:
        for a in i['attrs']:
            for sa in a:
                nw = len(list(sa.values())[0])
                sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    n_sents = sum(sent_lengths.values())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    for i in range(max_len + 1):
        n_len = sent_lengths.get(i, 0)
        print('%2d: %10d   %f%%' % (i, n_len, n_len * 100.0 / n_sents))
            
    return vocab


def encode_caption(imgs, stem_wtoi, attr_wtoi):
    """
    """
    n_imgs = len(imgs)
    n_stems = sum(len(img['stems']) for img in imgs)  # total number of captions

    stems_list = []
    image2stem = np.zeros(n_imgs, dtype='uint32')  # note: these will be one-indexed
    stem2image = np.zeros(n_stems, dtype='uint32')
    stem_attrs_list = []
    
    counter = 0
    for i, img in enumerate(imgs):
        n = len(img['stems'])
        assert n > 0, 'error: some image has no captions'

        image2stem[i] = counter
        for j, (stem, attr) in enumerate(zip(img['stems'], img['attrs'])):
            stem2image[counter] = i
            
            stem_wid = [stem_wtoi['START']]
            stem_wid.extend([stem_wtoi[w] if w in stem_wtoi else stem_wtoi['UNK'] for w in stem])
            stem_wid.append(stem_wtoi['EOS'])
            stems_list.append(stem_wid)

            stem_attrs = []
            for k, sa in enumerate(attr):
                attr_wid = [attr_wtoi['START']]
                attr_wid.extend([attr_wtoi[w] if w in attr_wtoi else attr_wtoi['UNK'] for w in list(sa.values())[0]])
                attr_wid.append(attr_wtoi['EOS'])
                stem_attrs.append(attr_wid)
            stem_attrs_list.append(stem_attrs)

            counter += 1
        
    return stems_list, image2stem, stem2image, stem_attrs_list

def coco_h5(params):
    imgs = json.load(open(os.path.join(data_root, 'coco_raw_with_attr.json'), 'r', encoding='utf8'))
    n_imgs = len(imgs)
    
    split = defaultdict(list)
    for i,img in enumerate(imgs):
        split[img['split']].append(i)
    new_imgs = [imgs[i] for i in split['val']]
    new_imgs.extend([imgs[i] for i in split['test']])
    new_imgs.extend([imgs[i] for i in split['train']])
    new_imgs.extend([imgs[i] for i in split['restval']])
    imgs = new_imgs
    del split
    del new_imgs

    # process image file by imread & imresize, and then save to dset
    dset = np.zeros((n_imgs, 256, 256, 3), dtype='uint8')
    for i, img in enumerate(imgs):
        I = smisc.imread(os.path.join(params['images_root'], img['filepath'], img['filename']))
        try:
            Ir = smisc.imresize(I, (256,256))
        except:
            print('failed resizing image %s ' % (img['filepath'] + '/' + img['filename'],))
            raise
        if len(Ir.shape) == 2:
            Ir = Ir[:, :, np.newaxis]
            Ir = np.concatenate((Ir,Ir,Ir), axis=2)
        dset[i] = Ir
        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, n_imgs, i*100.0/n_imgs))

    #skeleton vocab
    vocab_stem = build_vocab_stem(imgs, params)
    itow_stem = {i : w for i,w in enumerate(vocab_stem)} # a 1-indexed vocab translation table
    wtoi_stem = {w : i for i,w in enumerate(vocab_stem)} # inverse table
    json.dump(itow_stem, open(os.path.join(data_root, 'train/ix_to_word_stem.json'), 'w', encoding='utf8'))
    json.dump(wtoi_stem, open(os.path.join(data_root, 'train/word_to_ix_stem.json'), 'w', encoding='utf8'))

    #attribute vocab
    vocab_attr = build_vocab_attr(imgs, params)
    itow_attr = {i : w for i, w in enumerate(vocab_attr)}  # a 1-indexed vocab translation table
    wtoi_attr = {w : i for i, w in enumerate(vocab_attr)}
    json.dump(itow_attr, open(os.path.join(data_root, 'train/ix_to_word_attr.json'), 'w', encoding='utf8'))
    json.dump(wtoi_attr, open(os.path.join(data_root, 'train/word_to_ix_attr.json'), 'w', encoding='utf8'))


    stems_list, image2stem, stem2image, stem_attrs_list = encode_caption(imgs, wtoi_stem, wtoi_attr)
    label_cut, label_cut2 = image2stem[5000], image2stem[10000]
    idx_all = [img['imgid'] for img in imgs]

    split = 'val'
    with open(os.path.join(data_root, split, split+'_captions.pkl'), 'wb') as fd:
        pkl.dump([stems_list[label_cut:label_cut2], stem_attrs_list[:label_cut]], 
            fd, pkl.HIGHEST_PROTOCOL)
    out = h5py.File(os.path.join(data_root, split, split+'_images.h5'), 'w')
    out.create_dataset('image2stem', data=image2stem[:5000], dtype='uint32')
    out.create_dataset('stem2image', data=stem2image[:label_cut], dtype='uint32')
    out.create_dataset('images', data=dset[:5000], dtype='uint8')
    out.close()
    with open(os.path.join(data_root, split, split+'_idx.pkl'), 'wb') as fd:
        pkl.dump(idx_all[:5000], fd)
    
    split = 'test'
    with open(os.path.join(data_root, split, split+'_captions.pkl'), 'wb') as fd:
        pkl.dump([stems_list[label_cut:label_cut2], stem_attrs_list[label_cut:label_cut2]], 
            fd, pkl.HIGHEST_PROTOCOL)
    out = h5py.File(os.path.join(data_root, split, split+'_images.h5'), 'w')
    out.create_dataset('image2stem', data=image2stem[5000:10000] - label_cut, dtype='uint32')
    out.create_dataset('stem2image', data=stem2image[label_cut:label_cut2] - 5000, dtype='uint32')
    out.create_dataset('images', data=dset[5000:10000], dtype='uint8')
    out.close()
    with open(os.path.join(data_root, split, split+'_idx.pkl'), 'wb') as fd:
        pkl.dump(idx_all[5000:10000], fd)
    
    split = 'train'
    with open(os.path.join(data_root, split, split+'_captions.pkl'), 'wb') as fd:
        pkl.dump([stems_list[label_cut2:], stem_attrs_list[label_cut2:]], 
            fd, pkl.HIGHEST_PROTOCOL)
    out = h5py.File(os.path.join(data_root, split, split+'_images.h5'), "w")
    out.create_dataset('image2stem', data=image2stem[10000:] - label_cut2, dtype='uint32')
    out.create_dataset('stem2image', data=stem2image[label_cut2:] - 10000, dtype='uint32')
    out.create_dataset('images', data=dset[10000:], dtype='uint8')
    out.close()
    with open(os.path.join(data_root, split, split+'_idx.pkl'), 'wb') as fd:
        pkl.dump(idx_all[10000:], fd)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='E:\\WorkSpace\\Research\\NIC_SKEL\\data\\skeletonkey\\',
                        help='data root of skeletonkey model')
    parser.add_argument('--images_root', default='F:\\data\\nic\\mscoco\\',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--stem_word_count_thr', default=5, type=int, 
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--attr_word_count_thr', default=3, type=int, 
                        help='only words that occur more than this number of times will be put in vocab')                        
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    global data_root
    data_root = params['data_root']
    if not os.path.exists(data_root):
        os.mkdir(data_root)
    for val_name in ['train', 'val', 'test']:
        if not os.path.exists(os.path.join(data_root, val_name)):
            os.mkdir(os.path.join(data_root, val_name))

    #parsing_coco()
    # file 'caption_stem_attr_all.json' saved.
    
    #combine_result()
    # file 'coco_raw_with_attr.json' saved.

    # ship data into h5 data format.(only here images_root is used)
    #print('parsed input parameters:')
    coco_h5(params)
