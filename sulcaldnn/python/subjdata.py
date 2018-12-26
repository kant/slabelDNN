from scipy.io import loadmat, savemat
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from scipy.stats import mstats

def load_test_datan(test_image,labelnum,fsize,lsize,shift,ax):

    images1 = [loadmat(test_image)]
    count=1
    X =np.zeros((count,512,512,fsize))

    # create x and y location features
    nx, ny = (512, 512)
    x1 = np.linspace(0, 1, nx)
    y1 = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x1, y1)

    for i in range(count):
        img_a=images1[i]['img']
        dim1,dim2,dim3=img_a.shape
        
        img_a1=img_a[:,:,0]
        img_a1 = (img_a1 - np.min(img_a1)) / (np.max(img_a1) - np.min(img_a1))
        X[i,:,:,0]=img_a1
        
        
        X[i,:,:,1:dim3]=img_a[:,:,1:dim3]
        if fsize>dim3: ## 
            X[i,:,:,dim3]=xv
            X[i,:,:,dim3+1]=yv
    X2=np.reshape(X,[count, 512,512,fsize])

    # Adding ability to shift
    X2n=np.roll(X2,shift,axis=ax)

    return X2n

## To DO check the issue with below It was working but then some bug was introduced during refactoring
def load_valid_datan(valid_imagelist,valid_labellist,batch_size,labelnum,fsize,lsize,shift,ax):
    maxlength=len(valid_imagelist)

    nx, ny = (512, 512)
    x1 = np.linspace(0, 1, nx)
    y1 = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x1, y1)

    vn_start=0
    while 1:
        vn_end=vn_start+batch_size
        if vn_end>maxlength:
            vn_end=maxlength
        vcount=vn_end-vn_start
        images1 = [(loadmat(each)) for each in valid_imagelist.name[vn_start:vn_end]]
        labels1 = [loadmat(each) for each in valid_labellist.name[vn_start:vn_end]]
        vn_start=vn_end
        if vn_start>=maxlength:
            print("reset validation start count",vn_start)
            vn_start = 0
        X=np.zeros((vcount,512,512,fsize))
        y=np.zeros((vcount,512,512,lsize))

        for i in range(vcount):
            img_a=images1[i]['img']
            dim1,dim2,dim3=img_a.shape
            img_a1=img_a[:,:,0]
            img_a1 = (img_a1 - np.min(img_a1)) / (np.max(img_a1) - np.min(img_a1))
            X[i,:,:,0]=img_a1
  
            X[i,:,:,1:dim3]=img_a[:,:,1:dim3]
    
            if fsize>dim3:
                X[i,:,:,dim3]=xv
                X[i,:,:,dim3+1]=yv
  
            # Uncomment below if to check distmap normalization
            #img_a1=img_a[:,:,0]
            #img_a1 = (img_a1 - np.min(img_a1)) / (np.max(img_a1) - np.min(img_a1))
            #X[i,:,:,0]=img_a1
            x=labels1[i]['img']
            if labelnum == 100: ## Condition to combine all masks above 10 to single mask
                roi=ma.masked_where(x > 9, x)
            else:
                roi=ma.masked_where(x == labelnum, x)
            #roi=ma.masked_where(x == labelnum, x)
            labelled_img = np.multiply(1,roi.mask)
            labelled_img = np.asarray(labelled_img)
            y[i,:,:,0] = labelled_img
        X2=np.reshape(X,[vcount, 512,512,fsize])
        y2=np.reshape(y,[vcount, 512,512,lsize])
        #yield (X2, y2 )
        # Adding ability to shift
    X2n=np.roll(X2,shift,axis=ax)
    y2n=np.roll(y2,shift,axis=ax)

    return X2n, y2n
