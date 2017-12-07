import os, pprint, time, math
import numpy as np
from config import get_config
import matplotlib.pyplot as plt
import pandas as pd
import cv2 

pp = pprint.PrettyPrinter()
    
def main():

    #load configuration
    conf, _ = get_config()
    
    n_neighbors =5
    anal_dir=conf.log_dir+'anal/g_df/'
    df_km = pd.read_csv(anal_dir+'/'+str(n_neighbors)+'_AllKmeans.csv')
        
    
    def showKMPlt():       
        
        fig, ax = plt.subplots()  
         
        l_cluster = [None]*n_neighbors
        colors = ['red','pink','yellow','green','blue', 'brown', 'violet', 'orange', 'black', 'gray']
        for i in range(n_neighbors):
            cluster =df_km.ix[df_km.iloc[:,3] == i]
            cluster= np.asarray(cluster)
            l_cluster[i] = plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i])
         
        #for j in range(l_k.shape[0]):
        #    ax.annotate(l_k[j, 2], (l_k[j, 0],l_k[j, 1]))
        #ax.legend(loc=2)
        
        label = np.arange(n_neighbors)
        plt.legend(l_cluster, label)

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    def showAllKMPlt():       
        
        fig, ax = plt.subplots()  
         
        l_cluster = [None]*n_neighbors
        colors = ['red','pink','yellow','green','blue', 'brown', 'violet', 'orange', 'black', 'gray']
        for i in range(2):
            cluster =df_km.ix[df_km.iloc[:,2] == i]
            cluster= np.asarray(cluster)
            l_cluster[i] = plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i])
         
        #for j in range(l_k.shape[0]):
        #    ax.annotate(l_k[j, 2], (l_k[j, 0],l_k[j, 1]))
        #ax.legend(loc=2)
        
        label = np.arange(n_neighbors)
        plt.legend(l_cluster, label)

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()    
    
    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
     
        x1=np.abs(np.subtract(np.asarray(df_km.iloc[:,0]),event.xdata ))<0.1
        x2=np.abs(np.subtract(np.asarray(df_km.iloc[:,1]),event.ydata ))<0.1
        y= x1 & x2

        id = df_km.ix[y].iloc[:,4]
        img_path = list()
        with open(anal_dir+'xg_feature.csv','r') as file:    
            for line in file:
                x_id = line.split(',',2)
                if(float(x_id[0])in id.values):    
                    img_path.append(x_id[1])
        file.close()


        n_img = len(img_path)
        n_idx=1
        plt.figure(2)
        for f in img_path:
            f= f.replace('\\', '/')
            f = f.replace(' ','')
            f_img = cv2.imread(f,0)

            plt.subplot(1, n_img, n_idx)
            plt.imshow(f_img,'gray')
            n_idx+=1
        if n_img>0:
            plt.show()
        
    showAllKMPlt()
       

if __name__ == '__main__':
    main()
