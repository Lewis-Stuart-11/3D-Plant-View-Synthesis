![Overview of 3D reconstructions for one plant](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeGl2c2dvZXdjYjFxZWY3dWU2YWZhbTFzejd3YnNhdGhyaTN1cjAxayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xPaE9UFztwNhm0p2nX/giphy-downsized-large.gif)

# Overview 

This repository contains all the scripts for running our dataset, which is explained in our paper: 'High-fidelity wheat plant reconstruction using 3D Gaussian splatting and neural radiance fields'. 

The research paper can be accessed using this link: [GigaScience](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf022/8096368) 

The dataset can be accessed either on [Nottingham Plant Images](https://plantimages.nottingham.ac.uk/) or in the GigaScience repository, [GigaDB](https://doi.org/10.5524/102661)

In this experiment, we utilised two UR5 robots to capture images around a series of plants on a turntable. For each image that was captured, a transform was generated that depicts where the camera is in 3D space. This dataset is compatible with current view synthesis models, such as NeRF and 3DGS. We captured 20 different wheat plants each being imaged at 6 different time frames (due to a malfunction in our setup on the 2nd week of imaging, we were only able to capture 112 instances in total). At each captured instance, we trained a series of different NeRF and 3DGS models, and evaluated the outputs of the trained model. It is possible to run our models using NeRFStudio. Other models should be able to train our dataset, as long as the model follows the standard conventions for data input that are present in the original NeRF or 3DGS models.

A 'run_models.py' script can be executed that can view pre-trained models, train new view synthesis data, evaluate renders and export images/videos.

All results generated for the plants in this dataset can be viewed in the 'results.xlsx' document in our dataset.

![Overview of our entire dataset capturing process](https://i.imgur.com/MbREpJh.png)

# Running Our 3D Reconstructions

Firstly, download this repo: 
```git clone https://github.com/Lewis-Stuart-11/3D-Plant-View-Synthesis```

Next, the dataset should be downloaded and extracted from either [Nottingham Plant Images](https://plantimages.nottingham.ac.uk/datasets.html#3DGS_NeRF_Reconstruction) or [GigaDB](https://doi.org/10.5524/102661). Not all instances need to be downloaded and extracted, only the plants that you wish to execute. For ease of use, these should be downlaoded in the 'Plant_Dataset' directory in this repo.

To ensure that our models can be easily executed, we have included a 'run_models.py' script in this directory that is designed for simple interfacing with NeRFStudio as well as our dataset. **Running this script is required if you are executing our trained models**, as this script will correctly update the saved paths in the config file to match the dataset location on your local PC.

## Setting up NeRFStudio

In order to run the trained models, NeRFStudio must be installed correctly. NeRFStudio is an extremely robust framework for executing various view synthesis models, and they offer detailed documentation for how to run these various models. The installation process for NeRFStudio may seem complex, but in reality it is relatively straightforward. We recommend viewing the official installation page on the [NeRFStudio website](https://docs.nerf.studio/quickstart/installation.html). We suggest setting up an Anaconda environment to ensure that this model does not adversely impact other installed packages. Ensure that this is activated before running our scripts. 

It is important to note that our models have been tested to work with NeRFStudio version 1.1.3. We recommend installing this version if you need to run our already trained models, as later versions may not be compatible with these. If you are training from scratch, then any version should work.

## Executing the Models

Our script offers capabilities for simple interface with the NeRFStudio framework. This script can be controlled using command line arguments or by a config file. All the arguments can be displayed via the following command: **run_models.py --help**

Firstly, in order to set what scene instance to execute on, the name of the scene and the date for that plant instance must be provided- these are formatted as the ```--scene_dir``` and ```--date``` arguments (e.g. bc1_1033_3 & 06-03-24). If these are not set, then the script will iteratively load every plant instance in the dataset.

The main functionalities offered by our script are:

1) Viewing reconstructions: Since all the plant scenes in our dataset have been pre-trained, it is possible to load the saved checkpoints and view the final 3D reconstruction. The realtime renderer is automatically executed- this can be deactivated using the ```--skip_view``` argument.
2) Training: By setting the ```--train_model``` argument, the script will start training on the specified date scene. It is critical that the model you want to train with is set for training- this can be altered using the ```--model``` argument with the default being *nerfacto*. 
3) Evaluation: Once a scene has been trained, metric values can be generated using NeRFStudio or our masked PSNR generator. These can be set using the ```--eval``` and ```--eval_masked``` arguments.
4) Exports: Render pointclouds, meshes, images and videos of the trained scene.

## Examples

We offer several examples that can be executed using the following commands:

```run_models.py --config configs/view_3dgs.txt``` - This views a 3DGS model that has been trained on one of our plants

```run_models.py --config configs/train_3dgs.txt``` - This trains a new 3DGS model of one of our plants

```run_models.py --config configs/view_nerf.txt``` - This views a nerf model that has been trained on one of our plants

```run_models.py --config configs/train_nerf.txt``` - This trains a new nerf model of one of our plants

It is important that when executing this script that the command line, or other software for executing this script, is running from inside the directory where this repo is saved to. This ensures that the relative file paths are correct for these config files. Furthermore, these config files will only work if the dataset is downloaded inside the 'Plant_Dataset' directory in this repo.

We recommend viewing the contents of these files and execute them to see how our script works. These have been set to run on the bc1_1033_3 06-03-24 instance, and perform all the avaliable training, evaluation and exporting functionality. The real-time render will then be executed at the end for viewing the model. 

It is important to note that the training and evaluation of the model will be skipped, as the dataset already has pre-trained models and evluation data. To override this existing data, set the following argument: ```--override```.

## Training Your Own Data 

It is also possible to train on your own data. Our script should run effectively, provided that your data has the following directory structure:

```
+-- Scene 
|   +-- Date 
|   |   +-- images
|   |   |   +-- image_type
|   |   +-- transforms  
|   |   |   +-- transform_type
```

Both image_type and transform_type can be named anything, as long as this is set in the ```--img_type``` and ```--transform_type``` arguments.

It is important for gaussian splatting (colmap) data that the images are located inside the 'images' directory. This does not matter for standard nerf 'transform.json' files, as long as each transform's file path links to the correct image.

## Using Other Frameworks

It is entirely possible to train this dataset without using NeRFStudio. Firstly, ensure that whatever model you are using is correctly installed and compatible with standerdised NeRF and 3DGS datasets. 

If you are training on a NeRF representation (only requiring transforms) then the transform.json file can be executed. This can be found in the subdirectories in the 'transform' directory found in each of the scene date instances. For standard models, the transform file in the 'original', 'adjusted' (recommended) and 'segmented' directories should run correctly. If the model you are running also supports depth file paths, then the 'depth' transform file should run correctly. If the model you are running supports masks, then the 'mask' transform file should run correctly.

If you are training on a 3DGS representation (requiring point clouds) then the 'undistorted' subdirectory in the transforms directory can be executed. This contains a point cloud generated from COLMAP which is typically used for initialising the Gaussians in the environment. It is important to note that typically these models have an *-i* argument, which should be set to the directory containing the undistorted images to train on: *-i ../../images/undistorted*. To use the segmented images instead, set the directory path to *-i ../../images/undistorted_segmented*

# Dataset structure
<p>
    <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExODB3eW1nNWFwOWtmdXR2ZHB3bWtsZW1qNjE0MGkxYWh2dDZtYWM5MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/02aFHSuOrbebPdMKxk/giphy.gif" width="200" />
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

1) Images: Every captured RGB image is stored in this directory, along with generated: depth maps, segmented images and masks. Each of these types is stored in an individual directoy
2) Transforms: Contains all transforms for each of the images. Each subdirectory contains different transform information: 
    -  Original- transforms captured by our robot setup.
    -  Adjusted- transforms after bundle adjustment (**We recommend running these transforms for NeRF**).
    -  Depth- includes depth masks with the RGB images.
    -  Segmented- image file paths include segmented images, rather than the standard RGB images.
    -  Undistorted- Contains point cloud and transforms for undistorted images, all generated by Colmap. This does not contain a *transform.json* file, but the COLMAP output binary files (**We recommend running these transforms for 3DGS**).
3) Gaussian-Splatting: Stores all trained 3D gaussian splatting model data, including the evaluation metrics and exported .splat files. A .config file is included in each model and can be loaded using NeRFStudio, which will run the complete model locally after training.
4) NeRF: Stores all trained NeRF model data, including the evaluation metrics. A .config file is included in each model and can be loaded into NeRFStudio, which will run the complete model locally after training.
5) Colmap: Contains all SfM data generated in the bundle adjustment process for optimising our transforms. Each instance in our optimisation pipeline is stored in a separate directory, allowing inspection of how the camera poses were optimised at each stage of the process.
6) Exports: Stores all meshes, point clouds, rendered images and rendered videos for each trained model in each plant instance.

It is important to note that a groundtruth directory is included in date directories from the 19-03-24 to the 21-03-24. Each of these contain captured ground truth scans as well as the evaluation metrics for the point cloud reconstruction accuracy of that plant. Furthermore, a 'colmap-new-transforms' directory was generated on these dates which contains all colmap data for generating transforms from SfM.

Evaluation metrics that are generated by the NeRFStudio evaluation script are in a 'eval_*model*.json' (where model is the name of the NeRFStudio model that was being evaluated), which is stored in the same directory as the config files for the trained NeRF and 3DGS models. The masked psnr metrics that are calculated by our 'masked_psnr_calculator.py' script are stored as 'eval_masked.json' in the same directory.

We have a variety of different trained models on the different plant instances. We recommend running the 'adjusted (NeRF)' or 'undistorted (3DGS)' models, as these produced the most impressive results. 

The data for each trained model has the following path: 'plant_name/date/view_synth_type/experiment/model/1'. For clarity, the different subdirectories are:
1) view_synth_type- Either be nerf or gaussian-splatting
2) experiment- Name we set as the experiment for training (typically the directory name for the transform type)
3) model- the NeRFStudio model that was used for training

# Complimentory Repos

A lot of different components were developed in order to capture this dataset. These have been seperated into different repos, each performing a critical function for capturing this dataset. Each repo contains an individual README providing more information.

## 3DGS-to-PC

Once each of the 3DGS scenes had been trained, we required that this scene be converted into a point cloud. This repo can convert any 3DGS scene into a dense point cloud with impressive colouring and detail. This framework is useful for converting the 3DGS format, which typically requires a specialised renderer to view, to a point cloud format (which most 3D viewing software will support).

Link: https://github.com/Lewis-Stuart-11/3DGS-to-PC

![Showcase of 3DGS to PC for plant](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2EyMHdnN2RpOG1pa21oMWxzejFwaDF1OGxlbDh3cTl1dmZhdjkydCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/TLU72glPknJLuhal0B/giphy.gif)

## Multi-View Robotic Imaging

In order to train a view synthesis model on a plant, we needed a series of 2D images of the plant as well as the camera poses. We developed a view capture framework that can capture a series of views, equidistant from the centre of an object, using a ROS supported robot. A camera pose is recorded for each image, and these are all refined using a bundle adjustment process. The pipeline supports multiple robots as well as incorperating a turntable. Data is saved in a form compatible with training on modern view synthesis models and this was used for capturing the images and transforms for our dataset.

Link: https://github.com/Lewis-Stuart-11/Multi-View-Robotic-Imaging

![Showcase of duel robot setup](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExZG9idWU4M2drcGZjZWF3d2FjZGRlMmdnYTBhZ2c2ZDFydzVqNXk2ZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/2FMP6xCDFya9LhVXbs/giphy.gif)

## Dual UR5 Config

To control our robotic setup, we needed to accurately map the robots in ROS and include functionality for moving these both UR5 robots in parallel. This repo contains all UR5 URDF files for both robots, ensuring that both robots are correctly mapped in ROS. Also, we include MoveIt and config files, that allow both robots to be controlled individually, or in parallel. This package supports duel path planning with collision avoidance. Finally, we include various launch files, allowing the robots to be launched individually, or as a group. These robots can be executed in real life, or in a Gazebo simulation.

Link: https://github.com/Lewis-Stuart-11/Dual-UR5-Config

![Setup Comparison](https://i.imgur.com/sK5Dehf.png)

# Citation
```
@article{10.1093/gigascience/giaf022,
    author = {Stuart, Lewis A G and Wells, Darren M and Atkinson, Jonathan A and Castle-Green, Simon and Walker, Jack and Pound, Michael P},
    title = {High-fidelity wheat plant reconstruction using 3D Gaussian splatting and neural radiance fields},
    journal = {GigaScience},
    volume = {14},
    pages = {giaf022},
    year = {2025},
    month = {03},
    abstract = {The reconstruction of 3-dimensional (3D) plant models can offer advantages over traditional 2-dimensional approaches by more accurately capturing the complex structure and characteristics of different crops. Conventional 3D reconstruction techniques often produce sparse or noisy representations of plants using software or are expensive to capture in hardware. Recently, view synthesis models have been developed that can generate detailed 3D scenes, and even 3D models, from only RGB images and camera poses. These models offer unparalleled accuracy but are currently data hungry, requiring large numbers of views with very accurate camera calibration.In this study, we present a view synthesis dataset comprising 20 individual wheat plants captured across 6 different time frames over a 15-week growth period. We develop a camera capture system using 2 robotic arms combined with a turntable, controlled by a re-deployable and flexible image capture framework. We trained each plant instance using two recent view synthesis models: 3D Gaussian splatting (3DGS) and neural radiance fields (NeRF). Our results show that both 3DGS and NeRF produce high-fidelity reconstructed images of a plant subject from views not captured in the initial training sets. We also show that these approaches can be used to generate accurate 3D representations of these plants as point clouds, with 0.74-mm and 1.43-mm average accuracy compared with a handheld scanner for 3DGS and NeRF, respectively.We believe that these new methods will be transformative in the field of 3D plant phenotyping, plant reconstruction, and active vision. To further this cause, we release all robot configuration and control software, alongside our extensive multiview dataset. We also release all scripts necessary to train both 3DGS and NeRF, all trained models data, and final 3D point cloud representations. Our dataset can be accessed via https://plantimages.nottingham.ac.uk/ or https://https://doi.org/10.5524/102661. Our software can be accessed via https://github.com/Lewis-Stuart-11/3D-Plant-View-Synthesis.},
    issn = {2047-217X},
    doi = {10.1093/gigascience/giaf022},
    url = {https://doi.org/10.1093/gigascience/giaf022},
    eprint = {https://academic.oup.com/gigascience/article-pdf/doi/10.1093/gigascience/giaf022/62748311/giaf022.pdf},
}
```
