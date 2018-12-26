import tensorflow as tf
import numpy.ma as ma
from keras.layers import Input, Dense, Dropout, Activation, concatenate,Conv2D, MaxPooling2D, Flatten,Conv2DTranspose,BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, Adam, nadam, Adagrad, RMSprop
from keras import backend as K


#smooth = 0.001
#def dice_coef(y_true, y_pred):
#    y_true_f = K.flatten(y_true)
#    y_pred_f = K.flatten(y_pred)
#    intersection = K.sum(y_true_f * y_pred_f)
#    print(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#    print(intersection)
#    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#    #return (2. * (intersection + smooth)) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) # Fix

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
# Reference: https://github.com/keras-team/keras/issues/9395
def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2,3))
    numerator = K.sum(numerator)

    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2,3))
    denominator = K.sum(denominator)

    gen_dice_coef = 2*numerator/denominator

    return gen_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)# test

def get_unet(fsize,lsize,losstype,bn,lr):
    inputs = Input((512, 512, fsize))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    if bn: conv1 = BatchNormalization()(conv1) 
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    if bn: conv1 = BatchNormalization()(conv1) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(lsize, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    #Adam(lr=1e-5)
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    #keras.callbacks.EarlyStopping(monitor=dice_coef, min_delta=0, patience=0, verbose=0, mode='auto')
   # model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])
    if losstype>1:
        model.compile(optimizer=adam, loss=generalized_dice_loss, metrics=[generalized_dice_coeff])
        #model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])
        #model.compile(optimizer='Nadam', loss=dice_coef_loss, metrics=[dice_coef])
    if losstype==1:
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[dice_coef])

    print(model.summary())
    return model

