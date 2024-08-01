This repository contains all the scripts for running our dataset, which is explained in our paper: 'An Extensive High-fidelity Wheat Plant Reconstruction Dataset trained using 3D Gaussian Splatting and Neural Radiance Field'. 

The dataset can be accessed using this link: https://uniofnottm-my.sharepoint.com/my?id=%2Fpersonal%2Flewis%5Fstuart%5Fnottingham%5Fac%5Fuk%2FDocuments%2F3D%5FWheat%5FView%5FSynth%5FDataset%5FZipped&ga=1

In this experiment, we utilised two UR5 robots to capture images around a series of plants on a turntable. For each image that was captured, a transform was generated that depicts where the camera is in 3D space. This dataset is compatible with current view synthesis models, such as NeRF and 3DGS. We captured 20 different wheat plants each being imaged at 6 different time frames (due to a malfunction in our setup on the 2nd week of imaging, we were only able to capture 112 instances in total). At each captured instance, we trained a series of different NeRF and 3DGS models, and evaluated the outputs of the trained model. It is possible to run our models using NeRFStudio. Other models should be able to train our dataset, as long as the model follows the standard conventions for data input that are present in the original NeRF or 3DGS models.

An 'run_models.py' script can be run that can view pre-trained models, train new view synthesis data, evaluate and export images/videos

All results generated for the plants in this dataset can be viewed in the 'results.xlsx' document.

# Running Our 3D Reconstructions

Firstly, the dataset must be downloaded and extracted using the following link:  https://uniofnottm-my.sharepoint.com/my?id=%2Fpersonal%2Flewis%5Fstuart%5Fnottingham%5Fac%5Fuk%2FDocuments%2F3D%5FWheat%5FView%5FSynth%5FDataset%5FZipped&ga=1

To ensure that our models can be easily executed, we have included a 'run_models.py' script in this directory that is designed for simple interfacing with NeRFStudio as well as our dataset

## Setting up NeRFStudio

In order to run the trained models, NeRFStudio must be installed correctly. NeRFStudio is an extremely robust framework for executing various view synthesis models, and they offer detailed documentation for how to run these various models. The installation process for NeRFStudio may seem complex, but in reality it is relatively straightforward. We recommend viewing the official installation page on the NeRFStudio website: *https://docs.nerf.studio/quickstart/installation.html*. We suggest setting up an Anaconda environment to ensure that this model does not adversely impact other installed packages. Ensure that this is activated before running our scripts. 

It is important to note that our models are trained on version 1.0.3 (commit: 5fbb4b0af8b7ad60289b89043d8b79e9886f95c9). We recommend cloning this commit version when running our models, however, the latest version should still be compatible.

## Executing the Models

Our script offers capabilities for simple interface with the NeRFStudio framework. This script can be controlled using command line arguments or by a config file. All the arguments can be displayed via the following command: **run_models.py --help**

Firstly, in order to set what scene instance to execute on, the name of the scene and the date for that plant instance must be provided- these are formatted as the *scene_dir* and *date* arguments (e.g. bc1_1033_3 & ). If these are not set, then the script will iteratively load every plant instance in the dataset.

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

+-- Scene 
|   +-- Date 
|   |   +-- images
|   |   |   +-- image_type
|   |   +-- transforms  
|   |   |   +-- transform_type

Both image_type and transform_type can be named anything, as long as this is set in the *img_type* and *transform_type* arguments.

It is important for gaussian splatting (colmap) data that the images are located inside the 'images' directory. This does not matter for standard nerf 'transform.json' files, as long as each transform's file path links to the correct image.

## Using Other Frameworks

It is entirely possible to train this dataset without using NeRFStudio. Firstly, ensure that whatever model you are using is correctly installed and compatible with standerdised NeRF and 3DGS datasets. 

If you are training on a NeRF representation (only requiring transforms) then the transform.json file can be executed. This can be found in the subdirectories in the 'transform' directory found in each of the scene date instances. For standard models, the transform file in the 'original', 'adjusted' (recommended) and 'segmented' directories should run correctly. If the model you are running also supports depth file paths, then the 'depth' transform file should run correctly. If the model you are running supports masks, then the 'mask' transform file should run correctly.

If you are training on a 3DGS representation (requiring point clouds) then the 'undistorted' subdirectory in the transforms directory can be executed. This contains a point cloud generated from COLMAP which is typically used for initialising the gaussians in the environment. It is important to note that typically these models have an *-i* argument, which should be set to the directory containing the undistorted images to train on: *-i ../../images/undistorted*. To use the segmented images instead, set the directory path to *-i ../../images/undistorted_segmented*

# Data structure
Our dataset has the following structure:

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

# Citation
If you use this dataset or our accompanying scripts, please consider citing:

TO ADD