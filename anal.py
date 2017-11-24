import os, pprint, time
import numpy as np
import tensorflow as tf
from glob import glob
from random import shuffle
from model import generate, encode, decode
from utils import save_images, get_image
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
    npz_path ='C:/samples/img_download/wheels/wheeldesign/output/began2/17-11-14-11/'
    
    g_params = np.load( npz_path+'net_g.npz' )['params']
    d_params = np.load( npz_path+'net_d.npz' )['params']
    e_params = np.load( npz_path+'net_e.npz' )['params']
       
    saver = tf.train.import_meta_graph(npz_path+'began2_model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(npz_path))
    
    #col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
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
                    

    #z2 = tf.random_uniform((conf.n_batch, conf.n_img_out_pix,conf.n_img_out_pix,n_channel), minval=-1.0, maxval=1.0)
    z_fix =np.random.uniform(low=-1, high=1, size=(conf.n_batch, 64)).astype(np.float32)
    
    #load real image 
    data_files = glob(os.path.join(conf.data_dir,conf.dataset, "*"))
    shuffle(data_files)
    x_fix = data_files[0:conf.n_batch]
    x_fix=[get_image(f, conf.n_img_pix, is_crop=conf.is_crop, resize_w=conf.n_img_out_pix, is_grayscale = conf.is_gray) for f in x_fix]
    x_fix = np.array(x_fix).astype(np.float32)
    if(conf.is_gray == 1):
        s,h,w = x_fix.shape
        x_fix = x_fix.reshape(s,h, w,n_channel )    
    
    # run ae        
    x_im =sess.run(d_img,feed_dict={g_net:x_fix})  
    #x_img= x_img*255
    g_im =sess.run(g_img,feed_dict={z:z_fix})
    z_im =sess.run(d_img,feed_dict={e_net:z_fix})  
    #z_img= z_img*255
    
    # save image
    save_images(x_im, [n_grid_row,n_grid_row],os.path.join(checkpoint_dir, 'anal_AE_X.png'))
    save_images(g_im, [n_grid_row,n_grid_row],os.path.join(checkpoint_dir, 'anal_AE_G.png'))
    save_images(z_im, [n_grid_row,n_grid_row],os.path.join(checkpoint_dir, 'anal_AE_Z.png'))
      
    sess.close()

if __name__ == '__main__':
    main()
