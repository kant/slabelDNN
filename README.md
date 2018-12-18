# slabelDNN
"Improving Brain Sulcal Curve Labeling in Large Scale Cross-Sectional MRI using Deep Neural Networks"

Features used in the network include parcel labels, mean curvature, distance measure, spectral features 
as illustrated below for left hemisphere

<img src="https://github.com/MASILab/slabelDNN/blob/master/features_example.png" width="600px"/>

Output sulcal curves generated for each subject on left and right hemisphere

<img src="https://github.com/MASILab/slabelDNN/blob/master/outputs_example.png" width="600px"/>

## Quick Start
#### Get our docker image
```
sudo docker pull vuiiscci/sulcaldnn:deep_sulcalcurve_v1_0_0
```
#### Run Sulcal curve prediction
You can run the following command or change the "input_dir", then you will have the final segmentation results in output_dir
```
# you need to specify the input directory
export input_dir=/home/input_dir   
# make that directory
sudo mkdir $input_dir
# Test data 
Data: 
***
File structure:
* INPUTS
     - subjects.txt
     - subj1/
     - subj2/
     - subjn/

Each subject folder should have following files (Eg: for subj1)
* subj1
    - lh.target_image_GMimg_centralSurf.vtk
    - rh.target_image_GMimg_centralSurf.vtk
    - lh.target_image_GMimg_centralSurf.sphere.vtk
    - rh.target_image_GMimg_centralSurf.sphere.vtk
    - lh.target_image_GMimg_centralSurf.scurve
    - rh.target_image_GMimg_centralSurf.scurve
    - lh_parcel.txt
    - rh_parcel.txt
***
# set output directory
export output_dir=$input_dir/output
#run the docker
sudo nvidia-docker run -it --rm -v $input_dir:/INPUTS/ -v $output_dir:/OUTPUTS masidocker/spiders:deep_sulcalcurve_v1_0_0 /extra/run_sulcaldnn.sh

Output sulcal curves are generated as slabel files in Results folder and vtk files with labels for each subject on left and right hemisphere are saved in /OUTPUTS folder
```
## Detailed envrioment setting  

#### Testing platform
- Ubuntu 16.04
- cuda 8.0
- tensorflow 1.4.1
- Docker version 1.13.1-cs9
- Nvidia-docker version 1.0.1 to 2.0.3
```

#### install Docker
```
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce
```

#### install Nvidia-Docker
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
