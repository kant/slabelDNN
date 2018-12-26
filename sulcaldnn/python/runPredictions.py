
# coding: utf-8

import argparse
import pandas as pd
import Unet2D
import subjdata
from scipy.io import loadmat, savemat

import sys
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', help='where the model saved')
parser.add_argument('--python_code_path', help='python code dir')
parser.add_argument('--output_dir', help='output dir')

opt = parser.parse_args()
print(opt)
#sys.path.append('/home-local/parvatp/sulcaldnn-docker/extra/python')
model_dir=opt.model_dir
output_dir=opt.output_dir
sys.path.append(opt.python_code_path)


def getPredictedCurves(feature_data, subj, hemi, model_dir, output_dir):
    # initialize params
    fsize = 10
    lsize = 1
    losstype = 2  # 1- binary_crossentropy else dice_coef_loss
    learnrate = 0.0001
    shift = 0
    ax = 2

    if hemi == 'lh':
        labels = [1, 2, 3, 4, 5, 6, 8, 9]
    else:
        labels = [15, 16, 17, 18, 19, 20, 22, 23]

    for labelnum in labels:
        print(labelnum)
        weight_filepath = model_dir + "/epochinit_8_0001_loss_2label" + str(labelnum) + "_feature_10.hdf5"
        model_unet = Unet2D.get_unet(fsize, lsize, losstype, 1, learnrate)
        model_unet.load_weights(weight_filepath)

        X2_test = subjdata.load_test_datan(feature_data, labelnum, fsize, lsize, shift, ax)
        y2_pred = model_unet.predict(X2_test)
        savemat(output_dir + '/output_' + str(subj) + '_' + str(hemi) + '_' + str(labelnum) + '_spectra' + str(
            fsize) + '.mat', {'X2_test': X2_test, 'y2_pred': y2_pred})


subjects=pd.read_csv( output_dir + '/subjects.txt')

for subj in subjects:
    hemi='lh'
    print(subj)
    feature_data=output_dir + '/Planar/' + str(subj) + '_'+ str(hemi) + '_feature.mat'
    print(feature_data)
    getPredictedCurves(feature_data,subj,hemi,model_dir,output_dir)

    hemi = 'rh'
    feature_data = data=output_dir + '/Planar/' + str(subj) + '_' + str(hemi) + '_feature.mat'
    getPredictedCurves(feature_data, subj, hemi, model_dir, output_dir)




