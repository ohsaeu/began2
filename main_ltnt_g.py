import os, pprint, time
import numpy as np
import tensorflow as tf
from glob import glob
from random import shuffle
from model import generate, encode, decode, generateLatent
from utils import save_images, save_image, save_npz, get_image
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
    ##========================= DEFINE MODEL ===========================##
    z = tf.random_uniform(
                (conf.n_batch, conf.n_z), minval=-1.0, maxval=1.0)
    x_net =  tf.placeholder(tf.float32, [conf.n_batch, conf.n_img_pix, conf.n_img_pix, n_channel], name='real_images')
    k_t = tf.Variable(0., trainable=False, name='k_t')
    
    e_out_net, enc_vars, e_x_net = encode(x_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, is_train=False, reuse=False)
    d_x_net, dec_vars = decode(e_x_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel, is_train=False, reuse=False)
    
    g_out_net, g_vars, g_x_net = generateLatent(z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel,  is_train=True, reuse=False)
    d_g_net, _ = decode(g_x_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel, is_train=False, reuse=True)

    d_g_img=tf.clip_by_value((d_g_net + 1)*127.5, 0, 255)
    d_x_img=tf.clip_by_value((d_x_net + 1)*127.5, 0, 255)
    
    #d_vars = enc_vars + dec_vars

    g_loss = tf.reduce_mean(tf.abs(g_out_net - e_out_net))

    g_optim = tf.train.AdamOptimizer(conf.d_lr).minimize(g_loss, var_list=g_vars)

    summary_op = tf.summary.merge([
            tf.summary.image("AE_G", d_g_img),
            tf.summary.image("AE_x", d_x_img),
            tf.summary.scalar("loss/gloss", g_loss),
        ])

    # start session
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    # init directories
    checkpoint_dir = os.path.join(conf.log_dir,conf.curr_time)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # init summary writer for tensorboard
    summary_writer = tf.summary.FileWriter(checkpoint_dir,sess.graph)

    try:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(conf.load_dir, conf.ckpt_nm))
    except:
        pass

    data_files = glob(os.path.join(conf.data_dir,conf.dataset, "*"))
    shuffle(data_files)
    #sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(conf.sample_size, conf.z_dim)).astype(np.float32)
    # sample_seed = np.random.uniform(low=-1, high=1, size=(config.sample_size, z_dim)).astype(np.float32)

    ##========================= TRAIN MODELS ================================## 
    z_fix = np.random.uniform(-1, 1, size=(conf.n_batch, conf.n_z))

    x_fix = data_files[0:conf.n_batch]
    x_fix=[get_image(f, conf.n_img_pix, is_crop=conf.is_crop, resize_w=conf.n_img_out_pix, is_grayscale = conf.is_gray) for f in x_fix]
    x_fix = np.array(x_fix).astype(np.float32)

    x_fix = x_fix.reshape(x_fix.shape[0],x_fix.shape[1], x_fix.shape[2],n_channel )

    save_images(x_fix, [n_grid_row,n_grid_row],'{}/x_fix.png'.format(checkpoint_dir))

    cost_file = open(checkpoint_dir+ "/cost.txt", 'w', conf.n_buffer)
    n_step=0
    for epoch in range(conf.n_epoch):
        ## shuffle data
        shuffle(data_files)

        ## update sample files based on shuffled data
        '''
        sample_files = data_files[0:conf.sample_size]
        sample = [get_image(sample_file, conf.image_size, is_crop=conf.is_crop, resize_w=conf.output_size, is_grayscale = conf.is_gray) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        if(conf.is_gray == 1):
                s_size, s_height, s_width = sample_images.shape
                sample_images = sample_images.reshape(s_size,s_height, s_width,n_channel )
        print("[*] Sample images updated!")
        '''

        ## load image data
        n_iters = int(len(data_files)/conf.n_batch)

        for idx in range(0, n_iters):
           
            f_batch = data_files[idx*conf.n_batch:(idx+1)*conf.n_batch]
            data_batch = [get_image(f, conf.n_img_pix, is_crop=conf.is_crop, resize_w=conf.n_img_out_pix, is_grayscale = conf.is_gray) for f in f_batch]
            img_batch = np.array(data_batch).astype(np.float32)
            
            if conf.is_gray :
                s,h,w = img_batch.shape
                img_batch = img_batch.reshape(s, h, w, n_channel )
                
            
            #z_batch = np.random.normal(loc=0.0, scale=1.0, size=(conf.sample_size, conf.z_dim)).astype(np.float32)
            z_batch = np.random.uniform(low=-1, high=1, size=(conf.n_batch, conf.n_z)).astype(np.float32)

            fetch_dict = {
                "gopt":g_optim,
                "gloss": g_loss, 
            }
            if n_step % conf.n_save_log_step == 0:
                fetch_dict.update({
                    "summary": summary_op,
                })

            start_time = time.time()
            result = sess.run(fetch_dict, feed_dict={z:z_batch,x_net:img_batch})

            if n_step % conf.n_save_log_step == 0:
                summary_writer.add_summary(result['summary'], n_step)
                summary_writer.flush()

                gloss = result['gloss']

                cost_file.write("Epoch: ["+str(epoch)+"/"+str(conf.n_epoch)+"] ["+str(idx)+"/"+str(n_iters)+"] time: "+str(time.time() - start_time)+", g_loss: "+str(gloss)+ "\n")


            if n_step % conf.n_save_img_step == 0:
                #g_sample = sess.run(g_img, feed_dict={z: z_fix})
                g_ae, x_ae = sess.run([d_g_img,d_x_img] ,feed_dict={z:z_batch,x_net: img_batch})

                save_image(g_ae, os.path.join(checkpoint_dir,  '{}_AE_G.png'.format(n_step)))
                save_image(x_ae, os.path.join(checkpoint_dir, '{}_AE_X.png'.format(n_step)))
                saver.save(sess, os.path.join(checkpoint_dir,"temp_gltnt_began2_model.ckpt") )
                
            n_step+=1
        
        if epoch %conf.n_save_ckpt_epoch ==0: 
            '''   
            net_g_name = os.path.join(checkpoint_dir, str(n_step)+'_'+'net_g.npz')
            net_e_name = os.path.join(checkpoint_dir, str(n_step)+'_'+'net_e.npz')
            net_d_name = os.path.join(checkpoint_dir, str(n_step)+'_'+'net_d.npz')    
            save_npz(g_vars, name=net_g_name, sess=sess)
            save_npz(enc_vars, name=net_e_name, sess=sess)
            save_npz(dec_vars, name=net_d_name, sess=sess)
            '''
            saver.save(sess, os.path.join(checkpoint_dir,str(n_step)+"_"+"began2_model.ckpt") ) 
    '''
    net_g_name = os.path.join(checkpoint_dir, 'final_net_g.npz')
    net_e_name = os.path.join(checkpoint_dir, 'final_net_e.npz')
    net_d_name = os.path.join(checkpoint_dir, 'final_net_d.npz')          
    '''
    saver.save(sess, os.path.join(checkpoint_dir,"final_gltnt_began2_model.ckpt"))
    
    cost_file.close()
    
    sess.close()

if __name__ == '__main__':
    main()
