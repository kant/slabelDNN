#!/bin/bash

start=$(date +%s.%N)
# clean up
rm -r /OUTPUTS/*

# preprocessing for sulcal curves
/extra/sulcaldnn/run_run_sulcalDNN_preprocessing.sh /usr/local/MATLAB/MATLAB_Runtime/v92/
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "*******preprocessing time: %.6f seconds\n" $dur

start=$(date +%s.%N)
# generate deep segmentation
bash /extra/runPrediction.sh
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "*******segmentation time: %.6f seconds\n" $dur

start=$(date +%s.%N)
#postprocessing for sulcal curves
/extra/sulcaldnn/run_run_sulcalDNN_postprocessing.sh /usr/local/MATLAB/MATLAB_Runtime/v92/
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "*******postprocessing time: %.6f seconds\n" $dur

start=$(date +%s.%N)
#generate pdf 
#/extra/generate_light_PDF
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "*******generating pdf time: %.6f seconds\n" $dur

#clean up
# rm -r /OUTPUTS/Data_2D
# rm -r /OUTPUTS/DeepSegResults
# rm -r /OUTPUTS/dicom2nifti
# rm -r /OUTPUTS/FinalSeg
# rm -r /OUTPUTS/FinalResult/tmp
