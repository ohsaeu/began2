import os, pprint, time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from random import shuffle
from model import generate, encode, decode
from utils import save_image, get_image
from config import get_config
pp = pprint.PrettyPrinter()
    
def main():

    #load configuration
    conf, _ = get_config()
    pp.pprint(conf)
    if conf.is_gray :
        n_channel=1
    else:
        n_channel=3
    
    n_grid_row = int(np.sqrt(conf.n_batch))
    
    z = tf.random_uniform(
                (conf.n_batch, conf.n_z), minval=-1.0, maxval=1.0)
    # execute generator
    g_net,_ = generate(z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel, is_train=False, reuse=False) 
    # execute discriminator
    e_net,_ = encode(g_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, is_train=False, reuse=False)
    d_net,_ = decode(e_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel,is_train=False, reuse=False)
   
    g_img=tf.clip_by_value((g_net + 1)*127.5, 0, 255)
    d_img=tf.clip_by_value((d_net + 1)*127.5, 0, 255)
    # start session
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    # init directories
    checkpoint_dir = os.path.join(conf.log_dir,conf.curr_time)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # load and fetch variables
    npz_path ='C:/samples/img_download/wheels/data2/output/began2_data2_17-12-01-15-57/'
    itr ='390260_'
    
    g_params = np.load( npz_path+itr+'net_g.npz' )['params']
    d_params = np.load( npz_path+itr+'net_d.npz' )['params']
    e_params = np.load( npz_path+itr+'net_e.npz' )['params']
       
    saver = tf.train.import_meta_graph(npz_path+'began2_model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(npz_path))
    
    g_idx=0
    e_idx=0
    d_idx=0                
    for x in tf.trainable_variables():
        
        key = x.name.split(':')[0]
        scope = key.split('/')
        
        with tf.variable_scope(scope[0]) as vs1:
            vs1.reuse_variables()
            with tf.variable_scope(scope[1]) as vs2:
                vs2.reuse_variables()
                ref =tf.get_variable(scope[2], shape=x.shape) 
                if scope[0]=='generator' and g_idx<len(g_params):
                    ref1=tf.assign(ref,g_params[g_idx])
                    sess.run(ref1)
                    g_idx+=1
                elif scope[0]=='encoder' and e_idx<len(e_params):
                    ref1=tf.assign(ref,e_params[e_idx])
                    sess.run(ref1)
                    e_idx+=1
                elif scope[0]=='decoder' and d_idx<len(d_params):
                    ref1=tf.assign(ref,d_params[d_idx])
                    sess.run(ref1)
                    d_idx+=1
                    
    def manifoldG():
        z_test =np.random.uniform(low=-1, high=1, size=(conf.n_batch, 64)).astype(np.float32)
        #g_im =sess.run(g_img,feed_dict={z:z_test})
        step =int(len(z_test)/2) 
        for i in range(step):
            g_mnfd = [None]*64
            g_mnfd[0] = z_test[0]
            g_mnfd[63] = z_test[i+step]
            for j in range(1,63):
                g_mnfd[j] = g_mnfd[0]+ ((g_mnfd[63] -g_mnfd[0])/63 *j)
            g_intp =sess.run(g_img,feed_dict={z:np.asarray(g_mnfd)})
            save_image(g_intp, os.path.join(checkpoint_dir, str(i)+'mnfd_anal_G.png'))
    


    manifoldG()
       
    sess.close()

if __name__ == '__main__':
    main()
