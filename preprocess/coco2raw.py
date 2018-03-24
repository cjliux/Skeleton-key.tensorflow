#%%
import os
import sys
import json

def coco2raw(coco, coco_raw):
    with open(coco, 'r', encoding='utf8') as fd:
        dataset = json.load(fd)
    dataset = dataset['images']

    info_raw = []
    for img in dataset:
        new_img = {}
        for k, v in img.items():
            if k != 'sentences':
                new_img[k] = v
            else:
                new_img['captions'] = []
                for sent in v:
                    new_img['captions'].append(' '.join(sent['raw'].lower().split()).strip())
        info_raw.append(new_img)
            
    js = json.dumps(info_raw)
    with open(coco_raw,'w', encoding='utf8') as fd:
        fd.write(js)

if __name__=='__main__':
    try:
        data_path = sys.argv[1]
    except:
        data_path = 'E:\\WorkSpace\\Research\\NIC_SKEL\\data\\skeletonkey'
    
    coco2raw(coco=os.path.join(data_path,'dataset_coco.json'),
        coco_raw=os.path.join(data_path, 'coco_raw.json'))
        