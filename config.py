#-*- coding: utf-8 -*-
import argparse
import datetime

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

#flags
flags = add_argument_group('Flags')
# dataset 이름
flags.add_argument("--dataset", type=str, default= "201780112_128_gray")
# dataset 경로
flags.add_argument("--data_dir", type=str, default="C:/samples/img_download/wheels/data2/img/")
# 로그 파일 저장 이름
flags.add_argument("--log_dir", type=str, default="C:/samples/survey_data/g/began2_18-01-08-17-08_128_gray_20180103/")
# 체크포인트 로드 경로
flags.add_argument("--load_dir", type=str, default="C:/samples/survey_data/g/began2_18-01-08-17-08_128_gray_20180103/")
# 현재 시각
flags.add_argument("--curr_time", type=str, default=datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
# 체크포인트 저장 경로
flags.add_argument("--checkpoint_dir", type=str, default= "checkpoint")
# gamma for began
flags.add_argument('--gamma', type=float, default=0.7)
# lambda_k for began
flags.add_argument('--lambda_k', type=float, default=0.001)
# discriminator loss - learning rate
flags.add_argument('--d_lr', type=float, default=0.00008)
# generator loss - learning rate
flags.add_argument('--g_lr', type=float, default=0.00008)
# batch size
flags.add_argument("--n_batch", type=int, default=64)
# input image pixel
flags.add_argument("--n_img_pix", type=int, default=128)
# output image pixel
flags.add_argument("--n_img_out_pix", type=int, default=128)
# initial value for convolution 
flags.add_argument('--n_conv_hidden', type=int, default=128,choices=[64, 128],help='n in the paper')
# number of ecpochs
flags.add_argument("--n_epoch", type=int, default=25)
# initial variable dimension
flags.add_argument("--n_z", type=int, default=64)
# save log every # step
flags.add_argument("--n_save_log_step", type=int, default=1)
# save image every # step
flags.add_argument("--n_save_img_step", type=int, default=2)
# save checkpoint every # step
flags.add_argument("--n_save_ckpt_step", type=int, default=2)
# whether image is gray or color
flags.add_argument("--is_gray", type=str2bool, default=True)

# misc for analysis
flags.add_argument('--skip_ratio', type=float, default=1.6)
flags.add_argument('--skip_value', type=float, default=0.1)
flags.add_argument('--skip_epoch', type=float, default=0)
flags.add_argument('--delta', type=float, default=0.2)
flags.add_argument("--n_buffer", type=int, default=1)
flags.add_argument("--n_cluster", type=int, default=16)
flags.add_argument("--is_reload", type=str2bool, default=False)
flags.add_argument("--is_train", type=str2bool, default=True)
flags.add_argument("--is_crop", type=str2bool, default=True)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
