# coding: utf-8
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import tensorflow as tf
from model import ModelLevel1
from beam_search import CaptionGenerator
import config
import json
import utils
from utils import *
import h5py

ModeKeys = tf.contrib.learn.ModeKeys

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model_dir", None, """model directory""")
tf.flags.DEFINE_string("model_file", None, """model file to be used.""")
tf.flags.DEFINE_string("result_file", "result.json", """model file to be used.""")

def main(argv):
    assert FLAGS.model_dir, "model dir must be specified."

    # dirs
    if FLAGS.model_file is None:
        model_file = tf.train.latest_checkpoint(FLAGS.model_dir)
    else:
        model_file = os.path.join(FLAGS.model_dir, FLAGS.model_file)
    assert os.path.exists(model_file), """model file doesn't exists."""
        
    result_path = os.path.join(FLAGS.model_dir, "result")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    
    # model
    (test_stems_list, test_stem_attrs_list, test_images, 
        test_image2stem, test_stem2image, test_idx) = utils.load_coco_data(
                                        config.data_root, 'test', ret_idx=True)

    model = ModelLevel1(config, mode=ModeKeys.INFER)
    model.build()
    generator = CaptionGenerator(model, model.level1_word2ix, None,
                                beam_size_1level=5, beam_size_2level=None,
                                encourage_1level=0.0, encourage_2level=None,
                                level2=False)
    
    # session run
    result = []
    config_ = tf.ConfigProto(allow_soft_placement=True)
    config_.gpu_options.per_process_gpu_memory_fraction=0.6
    config_.gpu_options.allow_growth = True
    with tf.Session(config=config_) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.resotre(sess, model_file)

        for i, image in enumerate(test_images):
            print('***************')
            # images_batch contains only one image.
            image = utils.crop_image(image, False)
            top_pred = generator.beam_search(sess, image)

            print(i, test_idx[i], top_pred)

            # print GT
            #decoded = utils.decode_captions_2level(first_level_this, second_level_this, 
            #                    model.level1_model.idx_to_word, model.level2_model.idx_to_word)
            #print(decoded)

            result.append({'imgid': int(test_idx[i]), 'caption': top_pred})
    
    # save result.
    result_file = os.path.join(result_path, FLAGS.result_file)
    if not os.path.exists(result_file):
        os.rename(result_file, result_file+'.backup')
        print("Warning: " + result_file + " already exists and renamed to " + result_file + ".backup .")
    with open(result_file, 'w', encoding='utf8') as fd:
        json.dump(result, fd)


if __name__ == "__main__":
    tf.app.run()
