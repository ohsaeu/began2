import os, pprint, time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
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
    
    z = tf.random_uniform((conf.n_batch, conf.n_z), minval=-1.0, maxval=1.0)
    # execute generator
    g_net,_ = generate(z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel, is_train=False, reuse=False) 
    # execute discriminator
    e_net,_, _ = encode(g_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, is_train=False, reuse=False)
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

    #saver = tf.train.import_meta_graph(npz_path+'began2_model.ckpt.meta')
    #saver.restore(sess, tf.train.latest_checkpoint(npz_path))
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(conf.load_dir, conf.ckpt_nm))
    
    #load real image 
    data_files = glob(os.path.join(conf.data_dir,conf.dataset, "*"))
    shuffle(data_files)
    x_fix = data_files[0:conf.n_batch]
    x_fix=[get_image(f, conf.n_img_pix, is_crop=conf.is_crop, resize_w=conf.n_img_out_pix, is_grayscale = conf.is_gray) for f in x_fix]
    x_fix = np.array(x_fix).astype(np.float32)
    if(conf.is_gray == 1):
        s,h,w = x_fix.shape
        x_fix = x_fix.reshape(s,h, w,n_channel )   
         
    n_loop = 1
    
    def getRealAR():
        # run ae        
        x_im =sess.run(d_img,feed_dict={g_net:x_fix})  
        save_images(x_im, [n_grid_row,n_grid_row],os.path.join(checkpoint_dir, 'anal_AE_X.png'))
    
    def getRandomG():
     
        f_g = open(checkpoint_dir+ '/g_img.csv', 'a')
        
        # generate images from generator and ae
        for i in range(5):
            z_test =np.random.uniform(low=-1, high=1, size=(conf.n_batch, 64)).astype(np.float32)
            g_im =sess.run(g_img,feed_dict={z:z_test})  
            save_images(g_im, [n_grid_row,n_grid_row],os.path.join(checkpoint_dir, str(i)+'_anal_G.png'))
        #    g_im = g_im/127.5 - 1.
        #    ae_g_im =sess.run(d_img,feed_dict={g_net:g_im})  
        #    save_images(ae_g_im, [n_grid_row,n_grid_row],os.path.join(checkpoint_dir, str(i)+'_anal_AE_G.png'))
            
        
            for j in range(g_im.shape[0]):
                f_g.write(str(g_im[j].tolist()).replace("[", "").replace("]", "")+ '\n')
        f_g.close()
        
    def getFixedG(f_in):
        l_z = list()
        with open(f_in,'r') as file:    
            for line in file:
               l_z.append(np.fromstring(line, dtype=float, sep=','))
        file.close()
        n_loop = int(len(l_z)/64)
        
        l_z = np.asarray(l_z)
        
        for i in range(n_loop):
            fr = 64*i
            to = 64*(i+1)
            z_test =l_z[fr:to]
            g_im =sess.run(g_img,feed_dict={z:z_test})  
            save_images(g_im, [n_grid_row,n_grid_row],os.path.join(checkpoint_dir, '_anal_fix_G.png'))
            #g_im = g_im/127.5 - 1.
            #ae_g_im =sess.run(d_img,feed_dict={g_net:g_im})  
            #save_images(ae_g_im, [n_grid_row,n_grid_row],os.path.join(checkpoint_dir, str(i)+'_anal_AE_G.png'))
    def getRandomAE():
        # generate images from discriminator and ae
        for i in range(n_loop):
            z_test =np.random.uniform(low=-1, high=1, size=(conf.n_batch, conf.n_img_out_pix, conf.n_img_out_pix,n_channel)).astype(np.float32)
            d_im =sess.run(d_img,feed_dict={g_net:z_test})  
            save_images(d_im, [n_grid_row,n_grid_row],os.path.join(checkpoint_dir, str(i)+'_anal_D.png'))
        
    def saveFeatures():
        # get latent value from real images (10*n_batch)
        for i in range(n_loop):
            shuffle(data_files)
            f_test = data_files[0:conf.n_batch]
            x_test=[get_image(f, conf.n_img_pix, is_crop=conf.is_crop, resize_w=conf.n_img_out_pix, is_grayscale = conf.is_gray) for f in f_test]
            x_test = np.array(x_test).astype(np.float32)
            if(conf.is_gray == 1):
                s,h,w = x_test.shape
                x_test = x_test.reshape(s,h, w,n_channel ) 
        
            latent =sess.run(e_net,feed_dict={g_net:x_test}) 
            
            f_latent = open(checkpoint_dir+ '/latent.csv', 'a')
            for k in range(latent.shape[0]):
                f_latent.write(str(latent[k].tolist()).replace("[", "").replace("]", "")+ '\n')
            f_latent.close()
            
    def getFeatures():
        f_path=checkpoint_dir+'/latent.csv'#'C:/samples/img_download/wheels/wheeldesign/output/began2_anal/17-11-28-14-52/latent.csv'    
        data = pd.read_csv(f_path)
        
        n_latent = data.shape[1]
        mean = [None]*n_latent
        std = [None]*n_latent
        for i in range(n_latent):
            #i+=1
            latent = np.array(data.iloc[:, i:i+1])
            mean[i] = np.mean(latent)
            std[i] = np.std(latent)
            
        plt.show()
        return mean, std
    
    def generateFeature(mean, std):
        z_size = len(mean)
        feature = [None]*z_size
        for i in range(z_size):
            feature[i] = np.random.normal(loc=mean[i], scale=std[i], size=z_size*n_loop)
        return feature   
    
    
    def generateImage(feature):
        feature = np.array(feature)
        idx=0
        for i in range(n_loop):
            
            f_net = feature[:,idx:idx+64]
            f_img =sess.run(d_img,feed_dict={e_net:f_net}) 
            save_images(f_img, [n_grid_row,n_grid_row],os.path.join(checkpoint_dir, str(i)+'_anal_G_df.png'))
            idx+=64
    
    def getDiscMeanFeature(mean):
        mean = np.array(mean)
        mean = mean-2
       
        m_net = [None]*64
        for i in range(64):
            m_net[i] = mean +1/63 *i
        d_mnfd =sess.run(d_img,feed_dict={e_net:m_net})                       
        save_images(d_mnfd, [n_grid_row,n_grid_row],os.path.join(checkpoint_dir, 'anal_D_Mean_df.png'))    
    
    #getFixedG(conf.log_dir+'anal/g_df/z.csv')
    
    #getRealAR()
    getRandomG()
    #getRandomAE()
           
    #saveFeatures()
    #z_mean, z_std = getFeatures()
    #z_feature = generateFeature(z_mean, z_std)
    #shuffle(z_feature)
    #generateImage(z_feature)
    #getDiscMeanFeature(z_mean)
       
    sess.close()

if __name__ == '__main__':
    main()
