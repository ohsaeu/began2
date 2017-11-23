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
flags.add_argument("--dataset", type=str, default= "128_gray_rotate")
flags.add_argument("--data_dir", type=str, default="C:/samples/img_download/wheels/wheeldesign/img/rotate/")
flags.add_argument("--log_dir", type=str, default="C:/samples/img_download/wheels/wheeldesign/output/began2_anal/")
flags.add_argument("--curr_time", type=str, default=datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
flags.add_argument("--checkpoint_dir", type=str, default= "checkpoint")
#flags.add_argument("--sample_dir", type=str, default= "sample")
#flags.add_argument("--history_dir", type=str, default= "history")

flags.add_argument('--gamma', type=float, default=0.5)
flags.add_argument('--lambda_k', type=float, default=0.001)
flags.add_argument('--d_lr', type=float, default=0.00008)
flags.add_argument('--g_lr', type=float, default=0.00008)
#flags.add_argument("--beta1", type=float, default=0.5)

flags.add_argument('--n_conv_hidden', type=int, default=128,choices=[64, 128],help='n in the paper')
flags.add_argument("--n_epoch", type=int, default=25)
flags.add_argument("--n_z", type=int, default=64)

flags.add_argument("--n_batch", type=int, default=64)
flags.add_argument("--n_img_pix", type=int, default=128)
flags.add_argument("--n_img_out_pix", type=int, default=128)
flags.add_argument("--n_save_log_step", type=int, default=1)
flags.add_argument("--n_save_img_step", type=int, default=2)
flags.add_argument("--n_buffer", type=int, default=1)

flags.add_argument("--is_gray", type=str2bool, default=True)
flags.add_argument("--is_train", type=str2bool, default=True)
flags.add_argument("--is_crop", type=str2bool, default=False)
#flags.add_argument("--visualize", type=str2bool, default=False)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
