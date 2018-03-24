#data_root = 'E:\\WorkSpace\\Research\\NIC_SKEL\\data\\skeletonkey\\'
#images_root = 'F:\\data\\nic\\mscoco\\'
#model_root = 'E:\\WorkSpace\\Research\\NIC_SKEL\\code\\Skeleton-Key\\result\\'
data_root = '/home/zwruan/disk/Data/'
images_root = '/home/zwruan/disk/mscoco/'
model_root = '/home/zwruan/disk/WorkSpace/Skeleton-key/result/'


resnet_cpkt = '/home/zwruan/disk/tfslim/resnet_v1_152.ckpt'

# hyper params
LEVEL1_max_step = 16
LEVEL1_n_feats = 49
LEVEL1_dim_in_feat = 2048
LEVEL1_dim_feat = 512
LEVEL1_dim_embed = 512
LEVEL1_dim_hidden = 1024
LEVEL1_dim_factor=512
LEVEL1_conv_ksize=3
LEVEL1_alpha = 0.0
LEVEL1_dropout = True

LEVEL2_max_step = 6
LEVEL2_dim_feature = 2048
LEVEL2_dim_embed = 512
LEVEL2_dim_hidden = 1024
LEVEL2_dropout = True

# training details
batch_size = 32
n_epochs = 20
lrate = 0.0002
beta1 = 0.1
beta2 = 0.001
epsilon = 1e-8
clip_grad = 5.0

print_freq = 200
print_bleu = False

disp_freq = 10
save_freq = 200
valid_freq = 200
log_freq = 20

