![Overview of 3D reconstructions for one plant](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeGl2c2dvZXdjYjFxZWY3dWU2YWZhbTFzejd3YnNhdGhyaTN1cjAxayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xPaE9UFztwNhm0p2nX/giphy-downsized-large.gif)

# Quick Installation (For reviewers)

Firstly, download this repo, either using the download button or using the command: 

```git clone https://github.com/Lewis-Stuart-11/3D-Plant-View-Synthesis```

We have provided a Docker file for installation of NeRFStudio and automatic download of a single plant (with 5 date instances) from our dataset. This Docker file was tested on Ubuntu

Unfortunately, **this docker file will not work for all setups**, as NeRFStudio is a complicated software framework with many dependencies. We recommend performing the following steps instead:

1. Follow the NeRFStudio installation process via the official documentation *https://docs.nerf.studio/quickstart/installation.html* (we used anaconda for running NeRFStudio). We have validated the dataset on the latest version (1.1.3). The only potential issue with the latest version is installing GSplat (which runs 3DGS) failed when installing using NeRFStudio. However, we were able to install it directly via the Github repo: https://github.com/nerfstudio-project/gsplat.
2. Download a sample from our dataset via the following link: *https://cvl.cs.nott.ac.uk/resources/nerf_data/bc1_1033_3.zip*
3. Copy bc1_1033_3.zip to the same directory that this repo was downloaded to
4. Unzip bc1_1033_3.zip
5. Open a command line (that has NeRFStudio support) and cd to the repo directory

**Once installed**, run the following command:  

```run_models.py --config view_3DGS.txt```

This script will view an trained 3DGS model of the plant. If you would prefer to train and evaluate a new model of the same plant, run the following command:

```run_models.py --config train_3DGS.txt```

# Overview 

This repository contains all the scripts for running our dataset, which is explained in our paper: 'High-fidelity Wheat Plant Reconstruction using 3D Gaussian Splatting and Neural Radiance Fields'. 

The dataset can be accessed using this link: *https://uniofnottm-my.sharepoint.com/my?id=%2Fpersonal%2Flewis%5Fstuart%5Fnottingham%5Fac%5Fuk%2FDocuments%2F3D%5FWheat%5FView%5FSynth%5FDataset%5FZipped&ga=1*

In this experiment, we utilised two UR5 robots to capture images around a series of plants on a turntable. For each image that was captured, a transform was generated that depicts where the camera is in 3D space. This dataset is compatible with current view synthesis models, such as NeRF and 3DGS. We captured 20 different wheat plants each being imaged at 6 different time frames (due to a malfunction in our setup on the 2nd week of imaging, we were only able to capture 112 instances in total). At each captured instance, we trained a series of different NeRF and 3DGS models, and evaluated the outputs of the trained model. It is possible to run our models using NeRFStudio. Other models should be able to train our dataset, as long as the model follows the standard conventions for data input that are present in the original NeRF or 3DGS models.

An 'run_models.py' script can be run that can view pre-trained models, train new view synthesis data, evaluate and export images/videos

All results generated for the plants in this dataset can be viewed in the 'results.xlsx' document.

![Overview of our entire dataset capturing process](https://i.imgur.com/MbREpJh.png)

# Running Our 3D Reconstructions

Firstly, download this repo: 
```git clone https://github.com/Lewis-Stuart-11/3D-Plant-View-Synthesis```

Next, the dataset should be downloaded and extracted using the following link:  *https://uniofnottm-my.sharepoint.com/my?id=%2Fpersonal%2Flewis%5Fstuart%5Fnottingham%5Fac%5Fuk%2FDocuments%2F3D%5FWheat%5FView%5FSynth%5FDataset%5FZipped&ga=1*. Not all instances need to be downloaded and extracted, only  the plants that you wish to execute. For ease of use, these should be downlaoded in same directory as the repo.

To ensure that our models can be easily executed, we have included a 'run_models.py' script in this directory that is designed for simple interfacing with NeRFStudio as well as our dataset

## Setting up NeRFStudio

In order to run the trained models, NeRFStudio must be installed correctly. NeRFStudio is an extremely robust framework for executing various view synthesis models, and they offer detailed documentation for how to run these various models. The installation process for NeRFStudio may seem complex, but in reality it is relatively straightforward. We recommend viewing the official installation page on the NeRFStudio website: *https://docs.nerf.studio/quickstart/installation.html*. We suggest setting up an Anaconda environment to ensure that this model does not adversely impact other installed packages. Ensure that this is activated before running our scripts. 

It is important to note that our models have been tested to work with NeRFStudio version 1.1.3. We recommend installing this version if you need to run our already trained models, as later versions may not be compatible with these. If you are training from scratch, then any version should work.

## Executing the Models

Our script offers capabilities for simple interface with the NeRFStudio framework. This script can be controlled using command line arguments or by a config file. All the arguments can be displayed via the following command: **run_models.py --help**

Firstly, in order to set what scene instance to execute on, the name of the scene and the date for that plant instance must be provided- these are formatted as the *scene_dir* and *date* arguments (e.g. bc1_1033_3 & 06-03-24). If these are not set, then the script will iteratively load every plant instance in the dataset.

The main functionalities offered by our script are:

    - Viewing reconstructions: Since all the plant scenes in our dataset have been pre-trained, it is possible to load the saved checkpoints and view the final 3D reconstruction. The realtime renderer is automatically executed- this can be deactivated using the --skip_view argument.
    - Training: By setting the *--train_model* argument, the script will start training on the specified date scene. It is critical that the model you want to train with is set for training- this can be altered using the *--model* argument with the default being *nerfacto*. 
    - Evaluation: Once a scene has been trained, metric values can be generated using NeRFStudio or our masked PSNR generator. These can be set using the *--eval* and *--eval_masked* arguments.
    - Exports: Render pointclouds, meshes, images and videos of the trained scene.

## Examples

We offer two example config files: train_nerf.txt and train_3dgs.txt. We recommend viewing the contents of these files and execute them to see how our script works. To use these scripts, run: **run_models.py --config train_3dgs.txt** and **run_models.py --config train_nerf.txt**. These have been set to run on the bc1_1033_3 06-03-24 instance, and perform all the avaliable training, evaluation and exporting functionality. The real-time render will then be executed at the end for viewing the model. 

It is important to note that the training and evaluation of the model will be skipped, as the dataset already has pre-trained models and evluation data. To override this existing data, set the following argument: *--override*.

## Training Your Own Data 

It is also possible to train on your own data. Our script should run effectively, providing that your data has the following directory structure:

```
+-- Scene 
|   +-- Date 
|   |   +-- images
|   |   |   +-- image_type
|   |   +-- transforms  
|   |   |   +-- transform_type
```

Both image_type and transform_type can be named anything, as long as this is set in the *img_type* and *transform_type* arguments.

It is important for gaussian splatting (colmap) data that the images are located inside the 'images' directory. This does not matter for standard nerf 'transform.json' files, as long as each transform's file path links to the correct image.

## Using Other Frameworks

It is entirely possible to train this dataset without using NeRFStudio. Firstly, ensure that whatever model you are using is correctly installed and compatible with standerdised NeRF and 3DGS datasets. 

If you are training on a NeRF representation (only requiring transforms) then the transform.json file can be executed. This can be found in the subdirectories in the 'transform' directory found in each of the scene date instances. For standard models, the transform file in the 'original', 'adjusted' (recommended) and 'segmented' directories should run correctly. If the model you are running also supports depth file paths, then the 'depth' transform file should run correctly. If the model you are running supports masks, then the 'mask' transform file should run correctly.

If you are training on a 3DGS representation (requiring point clouds) then the 'undistorted' subdirectory in the transforms directory can be executed. This contains a point cloud generated from COLMAP which is typically used for initialising the gaussians in the environment. It is important to note that typically these models have an *-i* argument, which should be set to the directory containing the undistorted images to train on: *-i ../../images/undistorted*. To use the segmented images instead, set the directory path to *-i ../../images/undistorted_segmented*

# Data structure
<p>
    <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDhzdml0OHd3N2l6N2g4NjY4cGdtcXB2ZjJwcG50azN4dXh0Y2libyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/dK6aMOo6zmyYcF0ht6/giphy.gif" width="200" />
    <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExazZncHRiaGV5anNzaDU5ajRjdDE0YmVseW9oZXZ5aGljemtsbDV2eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/FlziTAHpKHQ4utpWGa/giphy.gif" width="200" />
    <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcHgzam9weDJjZXZxZmw0d3Vhb2hvaTU1Y3N2bWdobHdldHU2MXdnOSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Mpx3b9WI2YTWfI0byN/giphy.gif" width="200" />
    <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3JtZzV3c3Jwam9hdjZpbXNmbHN6M28zbHllY3RjNncwYWY3ejZ5ZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/W5O7f0cQTGRMpFscAc/giphy.gif" width="200" />
</p>

Our dataset has the following structure:

```
+-- Scene (plant) #1
|   +-- Date #1
|   |   +-- colmap  
|   |   +-- exports  
|   |   +-- gaussian-splatting  
|   |   +-- images  
|   |   +-- nerf  
|   |   +-- transforms  
|   +-- Date #2
|   ...
|   +-- Date #6
+-- Scene (plant) #2
...
+-- Scene (plant) #20
```

Each top level directory stores all information for each of the individual wheat plants, which we denote as a 'scene' for clarity. Every subdirectory in each scene directory stores all data collected on that specific date. Next, each date directoy is split into the following important subdirectories:

    1. Images: Every captured RGB image is stored in this directory, along with generated: depth maps, segmented images and masks. Each of these types is stored in an individual directoy
    2. Transforms: Contains all transforms for each of the images. Each subdirectory contains different transform information: 
        -  Original- transforms captured by our robot setup.
        -  Adjusted- transforms after bundle adjustment (**We recommend running these transforms for NeRF**).
        -  Depth- includes depth masks with the RGB images.
        -  Segmented- image file paths include segmented images, rather than the standard RGB images.
        -  Undistorted- Contains point cloud and transforms for undistorted images, all generated by Colmap. This does not contain a *transform.json* file, but the COLMAP output binary files (**We recommend running these transforms for 3DGS**).
    3. Gaussian-Splatting: Stores all trained 3D gaussian splatting model data, including the evaluation metrics and exported .splat files. A .config file is included in each model and can be loaded using NeRFStudio, which will run the complete model locally after training.
    4. NeRF: Stores all trained NeRF model data, including the evaluation metrics. A .config file is included in each model and can be loaded into NeRFStudio, which will run the complete model locally after training.
    5. Colmap: Contains all SfM data generated in the bundle adjustment process for optimising our transforms. Each instance in our optimisation pipeline is stored in a separate directory, allowing inspection of how the camera poses were optimised at each stage of the process.
    6. Exports: Stores all meshes, point clouds, rendered images and rendered videos for each trained model in each plant instance.

It is important to note that a groundtruth directory is included in date directories from the 19-03-24 to the 21-03-24. Each of these contain captured ground truth scans as well as the evaluation metrics for the point cloud reconstruction accuracy of that plant. Furthermore, a 'colmap-new-transforms' directory was generated on these dates which contains all colmap data for generating transforms from SfM.

Evaluation metrics that are generated by the NeRFStudio evaluation script are in a 'nerfstudio_eval.json', which is stored in the same directory as the config files for the trained NeRF and 3DGS models. The masked psnr metrics that are calculated by our 'masked_psnr_calculator.py' script are stored as 'masked_eval.json' in the same directory.

We have a variety of different trained models on the different plant instances. We recommend running the 'adjusted (NeRF)' or 'undistorted (3DGS)' models, as these produced the most impressive results. 

The data for each trained model has the following path: 'plant_name/date/view_synth_type/experiment/model/1'. For clarity, the different subdirectories are:
    - view_synth_type- Either be nerf or gaussian-splatting
    - experiment- Name we set as the experiment for training (typically the directory name for the transform type)
    - model- the NeRFStudio model that was used for training

# Complimentory Repos

A lot of different components were developed in order to capture this dataset. These have been seperated into different repos, each performing a critical function for capturing this dataset. Each repo contains an individual README providing more information.

## 3DGS-to-PC

Once each of the 3DGS scenes had been trained, we required that this scene be converted into a point cloud. This repo can convert any 3DGS scene into a dense point cloud with impressive colouring and detail. Ultimately, this is useful for converting the 3DGS format, which typically requires a specialised renderer to view, to a point cloud format (which most 3D viewing software will support).

Link: https://github.com/Lewis-Stuart-11/3DGS-to-PC

![Showcase of 3DGS to PC for plant](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2EyMHdnN2RpOG1pa21oMWxzejFwaDF1OGxlbDh3cTl1dmZhdjkydCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/TLU72glPknJLuhal0B/giphy.gif)

## Multi-View Robotic Imaging

In order to train a view synthesis model on a plant, we needed a series of 2D images of the plant as well as the camera poses. We developed a view capture framework that can capture a series of views, equidistant from the centre of an object, using a ROS supported robot. A camera pose is recorded for each image, and these are all refined using a bundle adjustment process. The pipeline supports multiple robots as well as incorperating a turntable. Data is saved in a form compatible with training on modern view synthesis models and this was used for capturing the images and transforms for our dataset.

![Showcase of duel robot setup](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdm5paXh0Z3lwb29pMGtyOWE1c2hzanFjZzV2emU4cGttazNleWEwdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/FE4JcF5CeUyELIEfiE/giphy-downsized-large.gif)

Link: https://github.com/Lewis-Stuart-11/Multi-View-Robotic-Imaging

## Duel UR5 Config

To control our robotic setup, we needed to accurately map the robots in ROS and include functionality for moving these both UR5 robots in parallel. This repo contains all UR5 URDF files for both robots, ensuring that both robots are correctly mapped in ROS. Next, we include MoveIt and config files, that allow both robots to be controlled individually, or in parallel. This package supports duel path planning with collision avoidance. Finally, we include various launch files, allowing the robots to be launched individually, or as a group. These robots can be executed in real life, or in a Gazebo simulation.

Link: https://github.com/Lewis-Stuart-11/Dual-UR5-Config

# Citation
If you use this dataset or our accompanying scripts, please consider citing:

TO ADD
