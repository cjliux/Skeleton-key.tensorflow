import os
import time

if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import tensorflow as tf

import config
import h5py
import utils
import pickle as pkl
from beam_search import CaptionGenerator
from model import HierarchicalModel

ModeKeys = tf.contrib.learn.ModeKeys

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("train_dir", None,
    "Directory for saving and loading model checkpoints. (rooted at model_root)")
tf.flags.DEFINE_string("checkpoint", None,
    "path pointing to pretrained model.(rooted at model_root)")

def main(argv):
    assert FLAGS.train_dir is not None, "train_dir is required"

    # model
    tf.logging.info('building model.')
    model = HierarchicalModel(config, mode=ModeKeys.TRAIN)
    loss = model.build()
    generator = CaptionGenerator(model, beam_size_1level=3, beam_size_2level=None,
                            encourage_1level=0.0, encourage_2level=None, level2=False)
    
    # train_op
    tf.logging.info('optimizer')
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=config.lrate, 
                            beta1=config.beta1, beta2=config.beta2, 
                            epsilon=config.epsilon)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='level1')
            level1_grads = tf.gradients(loss, optim_vars)
            grads_and_vars = [(i, j) for i, j in zip(level1_grads, optim_vars) if i is not None]
            grads_and_vars = [(tf.clip_by_value(grad, -config.clip_grad, config.clip_grad), var) for grad, var in grads_and_vars]
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
    
    # summary op
    tf.logging.info('summary op')
    tf.summary.scalar('batch_loss', loss)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    # for grad, var in grads_and_vars:
    #     tf.summary.histogram(var.op.name + '/gradient', grad)
    summary_op = tf.summary.merge_all()

    # data
    tf.logging.info('loading data...')
    (train_stems_list, train_stem_attrs_list, train_images, 
    train_image2stem, train_stem2image) = utils.load_coco_data(config.data_root, 'train')
    (val_stems_list, val_stem_attrs_list, val_images, 
    val_image2stem, val_stem2image) = utils.load_coco_data(config.data_root, 'val')
        
    # handling directories
    train_dir = os.path.join(config.model_root, FLAGS.train_dir)
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    log_dir = os.path.join(train_dir, 'log', 'train')
    if not tf.gfile.IsDirectory(log_dir):
        tf.logging.info("Creating log directory for training: %s", log_dir)
        tf.gfile.MakeDirs(log_dir)

    checkpoint = None
    if FLAGS.checkpoint is not None:
        checkpoint = os.path.join(config.model_root, FLAGS.checkpoint)
        assert os.path.exists(checkpoint), "checkpoint must exists if given."

    # stats:
    n_examples = len(train_stems_list)
    n_examples_val = len(val_stems_list)
    n_iters_per_epoch = int(np.ceil(float(n_examples) / config.batch_size))
    n_iters_val = int(np.ceil(float(n_examples_val) / config.batch_size))

    tf.logging.info("The number of epoch: " + str(config.n_epochs) + "\n"
                + "Data size: " + str(n_examples) + "\n"
                + "Batch size: " + str(config.batch_size) + "\n"
                + "Iterations per epoch: " + str(n_iters_per_epoch))

    # tf session
    config_ = tf.ConfigProto(allow_soft_placement = True)
    config_.gpu_options.per_process_gpu_memory_fraction = 0.6
    config_.gpu_options.allow_growth = True
    with tf.Session(config=config_) as sess:
        tf.global_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=40)

        # pretrained
        model.resnet.init_fn(sess)
        if checkpoint is not None: 
            print("Start training with checkpoint..")
            saver.restore(sess, checkpoint)
        
        # dynamic stats
        prev_loss_epo = np.inf
        curr_loss_epo = 0
        best_loss_val = np.inf
        curr_loss_val = 0
        i_global = 0

        start_t = time.time()
        for epo in range(config.n_epochs):
            # stochastic batching
            rand_idxs = list(np.random.permutation(n_examples))

            for it in range(n_iters_per_epoch):
                # next batch
                rand_idx = rand_idxs[it * config.batch_size : (it + 1) * config.batch_size]
                stems_batch, mask_batch = utils.list2batch([train_stems_list[i] for i in rand_idx])
                img_idx = train_stem2image[rand_idx]
                img_batch = utils.crop_image(train_images[img_idx], True)  

                feed_dict = {model.resnet.images: img_batch}
                img_features = sess.run(model.resnet.features, feed_dict)

                feed_dict = {model.level1_model.captions: stems_batch,
                             model.level1_model.mask: mask_batch,
                             model.level1_model.image_features: img_features,
                             model.level1_model.keep_prob: 0.5}
                _, loss_it = sess.run([train_op, loss], feed_dict)
                curr_loss_epo += loss_it

                # global iteration counts
                i_global += 1 

                # write summary for tensorboard visualization
                if it % config.log_freq == 0:
                    summary = sess.run(summary_op, feed_dict)
                    summary_writer.add_summary(summary, epo * n_iters_per_epoch + it)

                # periodical display 
                if it % config.disp_freq == 0:
                    tf.logging.info("[Train] E %d B %d C %.6f" % (epo, it, loss_it))

                if it % config.print_freq == 0:
                    ground_truths = stems_batch[0]
                    decoded = utils.decode_captions(ground_truths, model.level1_model.idx_to_word)
                    msg = "Ground truth: %s \n" % decoded
                    msg += str(ground_truths) + "\n"

                    predicted = generator.beam_search(sess, img_batch[0:1, :, :, :])
                    msg += "Generated caption: %s" % predicted
                    tf.logging.info(msg)

                # auto save 
                if i_global % config.save_freq == 0:
                    saver.save(sess, os.path.join(train_dir, 'model_level1_auto_save'), 
                            global_step=i_global)
                    tf.logging.info("model-auto-%s saved." % (i_global))

                # validate
                if i_global % config.valid_freq == 0:
                    cur_loss_val = 0
                    if config.print_bleu:
                        # TODO: some preparation for saving search result.
                        #all_gen_cap = np.ndarray((n_examples_val, 16))
                        pass

                    idxs_val = list(np.arange(n_examples_val))
                    for it_val in range(n_iters_val):
                        idx_val = idxs_val[it_val * config.batch_size:(it_val + 1) * config.batch_size]
                        stems_batch_val, mask_batch_val = utils.list2batch([val_stems_list[i] for i in idx_val])
                        img_idx_val = val_stem2image[idx_val]
                        img_batch_val = utils.crop_image(val_images[img_idx_val], False)

                        feed_dict_val = {model.resnet.images: img_batch_val}
                        img_features = sess.run(model.resnet.features, feed_dict_val)

                        feed_dict_val = {model.level1_model.captions: stems_batch_val,
                                         model.level1_model.mask: mask_batch_val,
                                         model.level1_model.image_features: img_features,
                                         model.level1_model.keep_prob: 1.0 }
                        curr_loss_val += sess.run(loss, feed_dict_val)

                        if config.print_bleu:
                            # TODO: beam search and evaluate bleu.
                            pass

                    curr_loss_val /= n_iters_val
                    tf.logging.info("[Valid] C %.6f"%curr_loss_val)

                    if curr_loss_val < best_loss_val:
                        best_loss_val = cur_loss_val
                        # better model
                        saver.save(sess, os.path.join(train_dir, 'model_level1_val'),
                                global_step=i_global)
                        tf.logging.info('model-val-%s saved.'%(i_global))
                    else:
                        # TODO: early stop checking.
                        pass
            # end for(i)
            curr_loss_epo /= n_iters_per_epoch

            # epoch summary:
            tf.logging.info("Previous epoch loss: " + str(prev_loss_epo) + "\n"
                        + "Current epoch loss: " + str(curr_loss_epo) + "\n"
                        + "Elapsed time: " + str(time.time() - start_t))
            prev_loss_epo = curr_loss_epo
            curr_loss_epo = 0

            # save model's parameters
            saver.save(sess, os.path.join(train_dir, 'model_level1_epo'), 
                    global_step=epo + 1)
            tf.logging.info("model-epo-%s saved." % (epo + 1))
                  
                # # print out BLEU scores and file write
                # if config.print_bleu:
                #     all_gen_cap = np.ndarray((n_examples_val, 16))
                #     for i in range(n_iters_val):
                #         features_batch = val_captions[i * config.batch_size:(i + 1) * config.batch_size]
                #         # feed_dict = {model.level1_model.features: features_batch}
                #         gen_cap = generator.beam_search(sess, features_batch)

                #         gen_cap = gen_cap[:16]
                #         all_gen_cap[i * config.batch_size:(i + 1) * config.batch_size, :len(gen_cap)] = gen_cap
                
                #     all_decoded = decode_captions(all_gen_cap, model.level1_model.idx_to_word)
                #     save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                    
                #     # TODO: evaluate() from pycocoevalcap
                #     scores = evaluate(data_path='./data', split='val', get_scores=True) 
                #     write_bleu(scores=scores, path=self.model_path, epoch=e)    
        
        # end for(e)
    # end with(sess)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
