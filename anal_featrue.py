import os, pprint, time, math
import numpy as np
import tensorflow as tf
from glob import glob
from random import shuffle
from model import generate, encode, decode
from utils import save_image, get_image
from config import get_config
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import cv2
import pandas as pd
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
        
    anal_dir=conf.log_dir+'anal/g_df/'
    # load and fetch variables
    '''
    npz_path =conf.log_dir
    itr ='222111_'
    
    n_neighbors =5
    anal_dir=conf.log_dir+'anal/g_df/'
    
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
    '''
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(conf.load_dir, conf.ckpt_nm))
                    
    def manifoldG():
        z_test =np.random.uniform(low=-1, high=1, size=(conf.n_batch, 64)).astype(np.float32)
        
        step =int(len(z_test)/2) 
        for i in range(step):
            g_mnfd = [None]*64
            g_mnfd[0] = z_test[i]
            g_mnfd[63] = z_test[i+step]
            for j in range(1,63):
                g_mnfd[j] = g_mnfd[0]+ ((g_mnfd[63] -g_mnfd[0])/63 *j)
            g_mnfd= np.asarray(g_mnfd)    
            g_intp =sess.run(g_img,feed_dict={z:g_mnfd})
            save_image(g_intp, os.path.join(checkpoint_dir, str(i)+'mnfd_anal_G.png'))
            
            f_g = open(checkpoint_dir+ '/g_mnfd.csv', 'a')
            for j in range(g_mnfd.shape[0]):
                f_g.write(str(g_mnfd[j].tolist()).replace("[", "").replace("]", "")+ '\n')
            f_g.close()
            
    def loadG(f_in):
        l_z = list()
        with open(f_in,'r') as file:    
            for line in file:
               l_z.append(np.fromstring(line, dtype=float, sep=','))
        file.close()
        
        idx = 110
        l_z = np.asarray(l_z)
        g_mnfd = [None]*64
        g_mnfd[0] = l_z[idx]
        g_mnfd[63] = l_z[idx+1]
        for j in range(1,63):
            g_mnfd[j] = g_mnfd[0]+ ((g_mnfd[63] -g_mnfd[0])/63 *j)
        g_mnfd= np.asarray(g_mnfd)    
        g_intp =sess.run(g_img,feed_dict={z:g_mnfd})
        save_image(g_intp, os.path.join(checkpoint_dir, str(idx)+'_detail_mnfd_anal_G.png'))
                   
    def manifoldD():
        f_data = glob(os.path.join(conf.data_dir,conf.dataset, "*"))
        shuffle(f_data)
        f_batch = f_data[0:conf.n_batch]
        x_test = [get_image(f, conf.n_img_pix, is_crop=conf.is_crop, resize_w=conf.n_img_out_pix, is_grayscale = conf.is_gray) for f in f_batch]
        x_test = np.array(x_test).astype(np.float32)
        if conf.is_gray :
                s,h,w = x_test.shape
                x_test = x_test.reshape(s, h, w, n_channel )
        step =int(conf.n_batch/2)
        for i in range(step):
            d_mnfd = [None]*64
            d_mnfd[0] = x_test[i]
            d_mnfd[63] = x_test[i+step]
            for j in range(1,63):
                d_mnfd[j] = d_mnfd[0]+ ((d_mnfd[63] -d_mnfd[0])/63 *j)
            d_intp =sess.run(d_img,feed_dict={g_net:d_mnfd})
            save_image(d_intp, os.path.join(checkpoint_dir, str(i)+'mnfd_anal_D.png'))
    
    def extractRealFeatrue():
        f_data = glob(os.path.join(conf.data_dir,conf.dataset, "*"))
        n_iters = int(len(f_data)/conf.n_batch)
        n_idx =1
        for idx in range(0, n_iters):
            f_batch = f_data[idx*conf.n_batch:(idx+1)*conf.n_batch]
            data_batch = [get_image(f, conf.n_img_pix, is_crop=conf.is_crop, resize_w=conf.n_img_out_pix, is_grayscale = conf.is_gray) for f in f_batch]
            img_batch = np.array(data_batch).astype(np.float32)
            
            if conf.is_gray :
                s,h,w = img_batch.shape
                img_batch = img_batch.reshape(s, h, w, n_channel )
                
            df_real =sess.run(e_net,feed_dict={g_net:img_batch}) 
            
            f_df_real = open(anal_dir+ '/x_feature.csv', 'a')
            for j in range(df_real.shape[0]):
                f_df_real.write(str(n_idx)+', '+f_batch[j]+ ', '+str(df_real[j].tolist()).replace("[", "").replace("]", "")+ '\n')
                n_idx+=1
            f_df_real.close()
    
    def generateGFeatrue():
        n_iters=20
        n_idx =1
        for _ in range(0, n_iters):
            z_df =np.random.uniform(low=-1, high=1, size=(conf.n_batch, 64)).astype(np.float32)
            z_net =sess.run(g_net,feed_dict={z:z_df})  
            z_img=np.clip((z_net + 1)*127.5, 0, 255)   
            df_z =sess.run(e_net,feed_dict={g_net:z_net}) 
            
            f_df_z = open(anal_dir+ 'g_feature.csv', 'a')
            f_z = open(anal_dir+ 'z.csv', 'a')
            for j in range(df_z.shape[0]):
                img_path = anal_dir+'G/'+str(n_idx)+'.jpg'
                cv2.imwrite(img_path, z_img[j])
                f_df_z.write(str(n_idx)+', '+img_path+ ', '+str(df_z[j].tolist()).replace("[", "").replace("]", "")+ '\n')
                f_z.write(str(z_df[j].tolist()).replace("[", "").replace("]", "")+ '\n')
                n_idx+=1
            f_df_z.close()
            f_z.close()
                    
    def doPCA(f_in, n_components):
        l_x = list()
        with open(f_in,'r') as file:    
            for line in file:
                pix = line.split(',',2)
                df = np.fromstring(pix[2], dtype=float, sep=',') 
                l_x.append(df)
                df =None
        file.close()

        l_p = PCA(n_components=n_components).fit(l_x)
        return l_p.transform(l_x) 
    
    def doKmeans(l_x):  
         
        kmeans = KMeans(init='k-means++', n_clusters=n_neighbors, n_init=10)
        kmeans.fit(l_x)
        n_idx=1
        f_km = open(anal_dir+'/'+str(n_neighbors)+'_AllKmeans.csv','w')
        for i in range(l_x.shape[0]):
            arr = np.concatenate((l_x[i],[i+1], [kmeans.labels_[i]]))     
            f_km.write(str(arr[0])+','+ str(arr[1])+ ','+str(arr[ 2])+ ','+str(arr[3])+','+str(n_idx)  +'\n')
            n_idx+=1
        f_km.close()    
        

    def saveClusterImages():
        df_km = pd.read_csv(anal_dir+'/'+str(n_neighbors)+'_Kmeans.csv')
        df_x = pd.read_csv(anal_dir+'/g_feature.csv')
        for i in range(n_neighbors):
            x_cluster =df_x.ix[df_km.iloc[:,3] == i]
            x_pix = x_cluster.iloc[:, 2:]
            x_pix = np.asarray(x_pix)
            
            x_total = x_pix.shape[0]
            c_net=None
            for j in range(math.ceil(x_total/64)):
                fr = 64*j
                to = 64*(j+1)
                if to >x_total:
                    c_net = np.ones((64,64))
                    to = x_total % 64
                    c_net[0:to, :] = x_pix[fr:x_total, :]
                else:
                    c_net = x_pix[fr:to, :]
                c_img =sess.run(d_img,feed_dict={e_net:c_net})
                save_image(c_img, anal_dir+'/'+str(i)+'_cluster_'+ str(j)+'.jpg')
    
    #loadG('C:/samples/img_download/celebA/output/2000_07_17-12-12-09-32/anal_17-12-28-10-16/g_mnfd.csv')       
    manifoldG()
    #manifoldD()
    #extractRealFeatrue()
    #generateGFeatrue()
    #doKmeans(doPCA(anal_dir+'xg_feature.csv',2)) #+'real_feature.csv'
    #saveClusterImages()
       
    sess.close()

if __name__ == '__main__':
    main()
