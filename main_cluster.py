import os, pprint, time
import numpy as np
import tensorflow as tf
import logging 
import logging.handlers
from glob import glob
from random import shuffle, randint
from model import generate, encode, decode
from utils import save_images, save_image, get_image
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

    # execute generator
    g_net, g_vars = generate(z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel,  is_train=True, reuse=False)
        
    # execute discriminator
    e_g_net, enc_vars,_ = encode(g_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, is_train=True, reuse=False)
    d_g_net, dec_vars = decode(e_g_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel, is_train=True, reuse=False)
    
    e_x_net, _,_ = encode(x_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, is_train=True, reuse=True)
    d_x_net, _ = decode(e_x_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel, is_train=True, reuse=True)
    
    g_img=tf.clip_by_value((g_net + 1)*127.5, 0, 255)
    #x_img=tf.clip_by_value((x_net + 1)*127.5, 0, 255)
    d_g_img=tf.clip_by_value((d_g_net + 1)*127.5, 0, 255)
    d_x_img=tf.clip_by_value((d_x_net + 1)*127.5, 0, 255)
    
    d_vars = enc_vars + dec_vars
    
    #d_x_img = tf.clip_by_value((d_x_net + 1)*127.5, 0, 255)
    #d_loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net_d_g, labels=tf.zeros_like(net_d_g)),name='d_loss_fake')
    #d_loss_x = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net_d_x, labels=tf.ones_like(net_d_x)),name='d_loss_real')
    #d_loss = d_loss_g + d_loss_x
    #g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net_g, labels=tf.ones_like(net_g)),name='g_loss')

    d_loss_g = tf.reduce_mean(tf.abs(d_g_net - g_net))
    d_loss_x = tf.reduce_mean(tf.abs(d_x_net - x_net))
    d_loss= d_loss_x - k_t * d_loss_g

    g_loss = tf.reduce_mean(tf.abs(d_g_net - g_net))

    g_optim = tf.train.AdamOptimizer(conf.g_lr).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(conf.d_lr).minimize(d_loss, var_list=d_vars)

    balance = conf.gamma * d_loss_x - g_loss
    measure = d_loss_x + tf.abs(balance)

    with tf.control_dependencies([d_optim, g_optim]):
        k_update = tf.assign(k_t, tf.clip_by_value(k_t + conf.lambda_k * balance, 0, 1))

    summary_op = tf.summary.merge([
            tf.summary.image("G", g_img),
            tf.summary.image("AE_G", d_g_img),
            tf.summary.image("AE_x", d_x_img),
            tf.summary.scalar("loss/dloss", d_loss),
            tf.summary.scalar("loss/d_loss_real", d_loss_x),
            tf.summary.scalar("loss/d_loss_fake", d_loss_g),
            tf.summary.scalar("loss/gloss", g_loss),
            #tf.summary.scalar("pd/pd_real", x_logits),
            #tf.summary.scalar("pd/pd_fake", g_logits),
            tf.summary.scalar("misc/m", measure),
            tf.summary.scalar("misc/kt", k_t),
            tf.summary.scalar("misc/balance", balance),
        ])

    # start session
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    # init directories
    checkpoint_dir = os.path.join(conf.log_dir,conf.curr_time)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    logger = logging.getLogger("log") 
    logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(os.path.join(checkpoint_dir,  'log.txt')) 
    logger.addHandler(fileHandler) 
    
    # init summary writer for tensorboard
    summary_writer = tf.summary.FileWriter(checkpoint_dir,sess.graph)

    saver = tf.train.Saver()
    if(conf.is_reload):
        saver.restore(sess, os.path.join(conf.load_dir, conf.ckpt_nm))


    data_files = list()
    data_num = list()
    total_num = 0
    for i in range(conf.n_cluster):
        data_loc = glob(os.path.join(conf.data_dir, conf.dataset, str(i), "*"))
        shuffle(data_loc)
        data_files.append(data_loc)
        num_data = len(data_loc)
        data_num.append(num_data)
        total_num += num_data
    
    #sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(conf.sample_size, conf.z_dim)).astype(np.float32)
    # sample_seed = np.random.uniform(low=-1, high=1, size=(config.sample_size, z_dim)).astype(np.float32)

    ##========================= TRAIN MODELS ================================## 
    z_fix = np.random.uniform(-1, 1, size=(conf.n_batch, conf.n_z))

    f_fix = list()
    for i in range(conf.n_cluster):
        f_fix.append(data_files[i][randint(0, data_num[i])])
    x_fix=[get_image(f, conf.n_img_pix, is_crop=conf.is_crop, resize_w=conf.n_img_out_pix, is_grayscale = conf.is_gray) for f in f_fix]
    x_fix = np.array(x_fix).astype(np.float32)

    x_fix = x_fix.reshape(x_fix.shape[0],x_fix.shape[1], x_fix.shape[2],n_channel )

    save_images(x_fix, [n_grid_row,n_grid_row],'{}/x_fix.png'.format(checkpoint_dir))

    #cost_file = open(checkpoint_dir+ "/cost.txt", 'w', conf.n_buffer)
    n_step=0
    for epoch in range(conf.n_epoch):

        ## load image data
        n_iters = int(total_num/conf.n_batch)

        for idx in range(0, n_iters):
           
            f_batch = list()
            for i in range(conf.n_cluster):
                f_batch.append(data_files[i][randint(0, data_num[i])])
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
            result = sess.run(fetch_dict, feed_dict={x_net:img_batch})
            m = result['m']

            if n_step % conf.n_save_log_step == 0:
                summary_writer.add_summary(result['summary'], n_step)
                summary_writer.flush()

                gloss = result['gloss']
                dloss = result['dloss']
                kt = result['kt']

                logger.info("Epoch: ["+str(epoch)+"/"+str(conf.n_epoch)+"] ["+str(idx)+"/"+str(n_iters)+"] time: "+str(time.time() - start_time)+", d_loss: "+str(dloss)+", g_loss:"+ str(gloss)+" measure: "+str(m)+", k_t: "+ str(kt)+ "\n")


            if n_step % conf.n_save_img_step == 0:
                g_sample, g_ae, x_ae = sess.run([g_img, d_g_img,d_x_img] ,feed_dict={x_net: x_fix})

                save_image(g_sample,os.path.join(checkpoint_dir, '{}_G.png'.format(n_step)))
                save_image(g_ae, os.path.join(checkpoint_dir,  '{}_AE_G.png'.format(n_step)))
                save_image(x_ae, os.path.join(checkpoint_dir, '{}_AE_X.png'.format(n_step)))
  
            n_step+=1
        
        saver.save(sess, os.path.join(checkpoint_dir,str(epoch)+"_"+str(n_step)+"_began2_model.ckpt") ) 

    saver.save(sess, os.path.join(checkpoint_dir,"final_began2_model.ckpt"))
    
    
    
    sess.close()

if __name__ == '__main__':
    main()
