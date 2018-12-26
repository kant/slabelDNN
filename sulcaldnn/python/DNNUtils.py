import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping,ModelCheckpoint, TensorBoard

import pandas as pd
import os
import keras
from scipy.io import savemat,loadmat
import matplotlib.pyplot as plt

from scipy.io import loadmat, savemat
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from scipy.stats import mstats

#Custom functions
import Unet2D
import subjdata4


hello = tf.constant('Hello, TensorFlow!')
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def trainData(t_imagelist,t_labellist,valid_imagelist,valid_labellist,test_imagelist,test_labellist,batch_size,labelnum,fsize,lsize,losstype,learnrate,num_epochs,outdir,init_weight_file,shift,ax):
    samp_per_epoch=len(t_imagelist)//batch_size
    validsamples_per_epoch=2*len(valid_imagelist)//batch_size

    #create model
    model_unet = Unet2D.get_unet(fsize,lsize,losstype,1,learnrate)    
    if init_weight_file:
        model_unet.load_weights(init_weight_file)
        weight_filename = "wdbc-weights_retrain-{epoch:04d}_dice-{val_generalized_dice_coeff:.2f}.hdf5"
    else:
        weight_filename = "wdbc-weights_epoch004-{epoch:04d}_dice-{val_generalized_dice_coeff:.2f}.hdf5"

    weight_dir = outdir+"/weights/weights_"+str(labelnum)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    # the filename will be populated at the end of each epoch, when the callback is called
    # values enclosed in curly braces will be substitued by Keras
    weight_filepath = os.path.join(weight_dir, weight_filename)
    checkpoint = ModelCheckpoint(weight_filepath, 
                         monitor='val_generalized_dice_coeff', 
                         verbose=0,)
    # the argument passed into the fit() function must be a list
    # if using multiple callbacks, you can use the .append() function or initialize them inline
    callbacks_list = [checkpoint]

    # Fit model
    history = model_unet.fit_generator(generator = subjdata3.load_train_datan(t_imagelist,t_labellist,batch_size,labelnum,fsize,lsize,shift,ax),
                    samples_per_epoch=samp_per_epoch,
                    validation_data = subjdata3.load_test_datan(valid_imagelist,valid_labellist,batch_size,labelnum,fsize,lsize,1,shift,ax),
                    validation_steps = validsamples_per_epoch,
                    callbacks=callbacks_list,
                    verbose=1,nb_epoch=num_epochs)
    # Save model
    model_unet.save(os.path.join(weight_dir,"epochinit_8_0001_loss_" + str(losstype) + "label" + str(labelnum) + "_feature_"+ str(fsize) +".hdf5"))
    dim1,dim2=test_labellist.shape
    # Predict test data
    X2_test,y2_test = subjdata3.load_test_datan(test_imagelist,test_labellist,dim1,labelnum,fsize,lsize,0,shift,ax)
    y2_pred=model_unet.predict(X2_test)
    output_dir = outdir+"/outputs/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    savemat(output_dir + '/output_0001_label' + str(labelnum) + '_spectra'+ str(fsize) +'.mat',{'X2_test':X2_test,'y2_test':y2_test,'y2_pred':y2_pred,'history':history.history})
    return history


def trainData2(t_imagelist,t_labellist,valid_imagelist,valid_labellist,test_imagelist,test_labellist,batch_size,labelnum,fsize,lsize,losstype,learnrate,num_epochs,outdir,init_weight_file,shift,ax):
    samp_per_epoch=len(t_imagelist)//batch_size
    validsamples_per_epoch=2*len(valid_imagelist)//batch_size

    #create model
    model_unet = Unet2D.get_unet(fsize,lsize,losstype,1,learnrate)    

    # If retraining
    if init_weight_file:
        model_unet.load_weights(init_weight_file)
        weight_filename = "wdbc-weights_retrain-{epoch:04d}_dice-{val_generalized_dice_coeff:.2f}.hdf5"
    else:
        weight_filename = "wdbc-weights_epoch004-{epoch:04d}_dice-{val_generalized_dice_coeff:.2f}.hdf5"

    weight_dir = outdir+"/weights/weights_"+str(labelnum)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    # the filename will be populated at the end of each epoch, when the callback is called
    # values enclosed in curly braces will be substitued by Keras
    weight_filepath = os.path.join(weight_dir, weight_filename)
    checkpoint = ModelCheckpoint(weight_filepath, 
                         monitor='val_generalized_dice_coeff', 
                         verbose=0,)
    # the argument passed into the fit() function must be a list
    # if using multiple callbacks, you can use the .append() function or initialize them inline
    callbacks_list = [checkpoint]

    # Fit model
    #history = model_unet.fit_generator(generator = subjdata2.load_train_datan(t_imagelist,t_labellist,batch_size,labelnum,fsize,lsize),
    history = model_unet.fit_generator(generator = subjdata4.load_train_datan(t_imagelist,t_labellist,batch_size,labelnum,fsize,lsize,shift,ax),
                    samples_per_epoch=samp_per_epoch,
                    #validation_data = subjdata2.load_test_datan(valid_imagelist,valid_labellist,batch_size,labelnum,fsize,lsize,1),
                    validation_data = subjdata4.load_test_datan(valid_imagelist,valid_labellist,batch_size,labelnum,fsize,lsize,1,shift,ax),
                    validation_steps = validsamples_per_epoch,
                    callbacks=callbacks_list,
                    verbose=1,nb_epoch=num_epochs)
    # Save model
    model_unet.save(os.path.join(weight_dir,"epochinit_8_0001_loss_" + str(losstype) + "label" + str(labelnum) + "_feature_"+ str(fsize) +".hdf5"))
    dim1,dim2=test_labellist.shape
    # Predict test data
    #X2_test,y2_test = subjdata2.load_test_datan(test_imagelist,test_labellist,dim1,labelnum,fsize,lsize,0)
    X2_test,y2_test = subjdata4.load_test_datan(test_imagelist,test_labellist,dim1,labelnum,fsize,lsize,0,shift,ax)
    y2_pred=model_unet.predict(X2_test)
    output_dir = outdir+"/outputs/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y2_pred2=np.roll(y2_pred,-shift,axis=ax)
    X2_test2=np.roll(X2_test,-shift,axis=ax)
    y2_test2=np.roll(y2_test,-shift,axis=ax)
    savemat(output_dir + '/output_0001_label' + str(labelnum) + '_spectra'+ str(fsize) +'.mat',{'X2_test':X2_test2,'y2_test':y2_test2,'y2_pred':y2_pred2,'history':history.history})
    return history

def trainData3(t_imagelist,t_labellist,valid_imagelist,valid_labellist,test_imagelist,test_labellist,batch_size,labelnum,fsize,lsize,losstype,learnrate,num_epochs,outdir,init_weight_file,shift,ax,refdata):
    samp_per_epoch=len(t_imagelist)//batch_size
    validsamples_per_epoch=2*len(valid_imagelist)//batch_size

    #create model
    model_unet = Unet2D.get_unet(fsize,lsize,losstype,1,learnrate)    

    # If retraining
    if init_weight_file:
        model_unet.load_weights(init_weight_file)
        weight_filename = "wdbc-weights_retrain-{epoch:04d}_dice-{val_generalized_dice_coeff:.2f}.hdf5"
    else:
        weight_filename = "wdbc-weights_epoch004-{epoch:04d}_dice-{val_generalized_dice_coeff:.2f}.hdf5"

    weight_dir = outdir+"/weights/weights_"+str(labelnum)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    # the filename will be populated at the end of each epoch, when the callback is called
    # values enclosed in curly braces will be substitued by Keras
    weight_filepath = os.path.join(weight_dir, weight_filename)
    checkpoint = ModelCheckpoint(weight_filepath, 
                         monitor='val_generalized_dice_coeff', 
                         verbose=0,)
    # the argument passed into the fit() function must be a list
    # if using multiple callbacks, you can use the .append() function or initialize them inline
    callbacks_list = [checkpoint]

    # Fit model
    #history = model_unet.fit_generator(generator = subjdata2.load_train_datan(t_imagelist,t_labellist,batch_size,labelnum,fsize,lsize),
    history = model_unet.fit_generator(generator = subjdata5.load_train_datan(t_imagelist,t_labellist,batch_size,labelnum,fsize,lsize,shift,ax,refdata),
                    samples_per_epoch=samp_per_epoch,
                    #validation_data = subjdata2.load_test_datan(valid_imagelist,valid_labellist,batch_size,labelnum,fsize,lsize,1),
                    validation_data = subjdata5.load_test_datan(valid_imagelist,valid_labellist,batch_size,labelnum,fsize,lsize,1,shift,ax,refdata),
                    validation_steps = validsamples_per_epoch,
                    callbacks=callbacks_list,
                    verbose=1,nb_epoch=num_epochs)
    # Save model
    model_unet.save(os.path.join(weight_dir,"epochinit_8_0001_loss_" + str(losstype) + "label" + str(labelnum) + "_feature_"+ str(fsize) +".hdf5"))
    dim1,dim2=test_labellist.shape
    # Predict test data
    #X2_test,y2_test = subjdata2.load_test_datan(test_imagelist,test_labellist,dim1,labelnum,fsize,lsize,0)
    X2_test,y2_test = subjdata5.load_test_datan(test_imagelist,test_labellist,dim1,labelnum,fsize,lsize,0,shift,ax,refdata)
    y2_pred=model_unet.predict(X2_test)
    output_dir = outdir+"/outputs/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y2_pred2=np.roll(y2_pred,-shift,axis=ax)
    X2_test2=np.roll(X2_test,-shift,axis=ax)
    y2_test2=np.roll(y2_test,-shift,axis=ax)
    savemat(output_dir + '/output_0001_label' + str(labelnum) + '_spectra'+ str(fsize) +'.mat',{'X2_test':X2_test2,'y2_test':y2_test2,'y2_pred':y2_pred2,'history':history.history})
    return history
