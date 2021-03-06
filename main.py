import os, pprint, time
import numpy as np
import tensorflow as tf
from glob import glob
from random import shuffle
from model import generate, encode, decode
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
 
    # define generator
    g_net, g_vars = generate(z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel,  is_train=True, reuse=False)
        
    # define discriminator
    e_g_net, enc_vars,_ = encode(g_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, is_train=True, reuse=False)
    d_g_net, dec_vars = decode(e_g_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel, is_train=True, reuse=False)
    e_x_net, _,_ = encode(x_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, is_train=True, reuse=True)
    d_x_net, _ = decode(e_x_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel, is_train=True, reuse=True)
    
    # image de-normalization
    g_img=tf.clip_by_value((g_net + 1)*127.5, 0, 255)
    d_g_img=tf.clip_by_value((d_g_net + 1)*127.5, 0, 255)
    d_x_img=tf.clip_by_value((d_x_net + 1)*127.5, 0, 255)
    
    d_vars = enc_vars + dec_vars

    # define discriminator and generator losses
    d_loss_g = tf.reduce_mean(tf.abs(d_g_net - g_net))
    d_loss_x = tf.reduce_mean(tf.abs(d_x_net - x_net))
    d_loss= d_loss_x - k_t * d_loss_g
    g_loss = tf.reduce_mean(tf.abs(d_g_net - g_net))

    # define optimizer
    d_optim = tf.train.AdamOptimizer(conf.d_lr).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(conf.g_lr).minimize(g_loss, var_list=g_vars)

    balance = conf.gamma * d_loss_x - g_loss
    measure = d_loss_x + tf.abs(balance)

    with tf.control_dependencies([d_optim, g_optim]):
        k_update = tf.assign(k_t, tf.clip_by_value(k_t + conf.lambda_k * balance, 0, 1))

    # define summary for tensorboard
    summary_op = tf.summary.merge([
            tf.summary.image("G", g_img),
            tf.summary.image("AE_G", d_g_img),
            tf.summary.image("AE_x", d_x_img),
            tf.summary.scalar("loss/dloss", d_loss),
            tf.summary.scalar("loss/d_loss_real", d_loss_x),
            tf.summary.scalar("loss/d_loss_fake", d_loss_g),
            tf.summary.scalar("loss/gloss", g_loss),
            tf.summary.scalar("misc/m", measure),
            tf.summary.scalar("misc/kt", k_t),
            tf.summary.scalar("misc/balance", balance),
        ])

    # start session
    sess = tf.InteractiveSession()#config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    # init directories
    checkpoint_dir = os.path.join(conf.log_dir,conf.curr_time)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # init summary writer for tensorboard
    summary_writer = tf.summary.FileWriter(checkpoint_dir,sess.graph)

    saver = tf.train.Saver()
    if(conf.is_reload):
        saver.restore(sess, os.path.join(conf.load_dir, conf.ckpt_nm))

    # load real image info and shuffle them
    data_files = glob(os.path.join(conf.data_dir,conf.dataset, "*"))
    shuffle(data_files)

    # save real fixed image
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

        ## load image data
        n_iters = int(len(data_files)/conf.n_batch)

        for idx in range(0, n_iters):
            # make image batch
            f_batch = data_files[idx*conf.n_batch:(idx+1)*conf.n_batch]
            data_batch = [get_image(f, conf.n_img_pix, is_crop=conf.is_crop, resize_w=conf.n_img_out_pix, is_grayscale = conf.is_gray) for f in f_batch]
            img_batch = np.array(data_batch).astype(np.float32)
            
            if conf.is_gray :
                s,h,w = img_batch.shape
                img_batch = img_batch.reshape(s, h, w, n_channel )
                
            fetch_dict = {
                "kupdate": k_update,
                "m": measure,
            }
            if n_step % conf.n_save_log_step == 0:
                fetch_dict.update({
                    "summary": summary_op,
                    "gloss": g_loss,
                    "dloss": d_loss,
                    "kt": k_t,
                })

            start_time = time.time()
            # run the session!
            result = sess.run(fetch_dict, feed_dict={x_net:img_batch})
            
            # get the result
            m = result['m']

            if n_step % conf.n_save_log_step == 0:
                summary_writer.add_summary(result['summary'], n_step)
                summary_writer.flush()

                # write cost to a file
                gloss = result['gloss']
                dloss = result['dloss']
                kt = result['kt']
                cost_file.write("Epoch: ["+str(epoch)+"/"+str(conf.n_epoch)+"] ["+str(idx)+"/"+str(n_iters)+"] time: "+str(time.time() - start_time)+", d_loss: "+str(dloss)+", g_loss:"+ str(gloss)+" measure: "+str(m)+", k_t: "+ str(kt)+ "\n")

            # save generated image file
            if n_step % conf.n_save_img_step == 0:

                g_sample, g_ae, x_ae = sess.run([g_img, d_g_img,d_x_img] ,feed_dict={x_net: x_fix})

                save_image(g_sample,os.path.join(checkpoint_dir, '{}_G.png'.format(n_step)))
                save_image(g_ae, os.path.join(checkpoint_dir,  '{}_AE_G.png'.format(n_step)))
                save_image(x_ae, os.path.join(checkpoint_dir, '{}_AE_X.png'.format(n_step)))
                
            n_step+=1

        # save checkpoint    
        saver.save(sess, os.path.join(checkpoint_dir,str(epoch)+"_"+str(n_step)+"_began2_model.ckpt") ) 

    # save final checkpoint  
    saver.save(sess, os.path.join(checkpoint_dir,"final_began2_model.ckpt"))
    
    cost_file.close()
    
    sess.close()

if __name__ == '__main__':
    main()
