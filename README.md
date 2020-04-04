# DISCo: Deep learning, Instance Segmentation and Correlations for cell segmentation in calcium imaging


This is a method to perform the cell segmentaiton step in caclium imaging analysis, which uses the temporal information from caclium imaging videos in form of correlations, and combines a deep learning model with an instance segmentation algorithm. 

## Publication

**"DISCo: Deep learning, Instance Segmentation, and Correlations for cell segmentation in calcium imaging"**, E. Kirschbaum, A. Bailoni, F. A. Hamprecht, *arXiv preprint arXiv:1908.07957*, 2019. 
[[pdf]](https://arxiv.org/pdf/1908.07957.pdf)

## Requirements:

* [**Python 3.6 (or later)**](https://www.python.org/): we recommend installing it with [Anaconda](https://www.anaconda.com/download/)
* [**PyTorch 1.0 (or later)**](http://pytorch.org/)
* [**GASP**](https://github.com/abailoni/GASP)
* [**inferno 0.3 (or later)**](https://github.com/inferno-pytorch/inferno)

## Preparations 

1. Download or clone this repository
2. Install GASP as described [here](https://github.com/abailoni/GASP)
3. Get inferno as described [here](https://github.com/inferno-pytorch/inferno)
4. Download the neurofinder training and test data from [here]()
5. Extract the neurofinder data into HDF5 files:   
	* create for each neurofinder video a HDF5 file with a dataset named 'video' containing the video with shape time x X x Y    
	* create a file named BF_labels.h5 containing the foreground-background labels for each video   
	* create a file summary_images.h5 containing the mean intensity projection for each video   
	* create a file gt_segmentations.h5 containing the instance labels for each video   

## Usage

| **Option** | **Name** | **Description** |  
|--------|-----|-----------|   
| `-p` | path | Path to the folder containing the .h5 video files and the ground truth segmentations  |   
| `-m` | mode | Decide whether a single network is trained on all videos ('disco') or individual networks on the five dataset series ('discos') |    
| `-gpu` | gpu ID | Select the GPU to train on. |   
| `-a` | additional ending | Additional ending to the output filename. |   


Example:   
` python run.py -p ../neurofinder_videos/ -m disco -gpu 1`   
