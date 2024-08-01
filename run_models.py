import os
import json
import copy
import datetime
import subprocess
import signal
import shutil
import configargparse

from psnr_calculator import calculate_masked_psnr_set

GAUSSIAN_SPLATTING_MODELS = ["splatfacto"]

"""Returns a list of subdirectories in the given directory path."""
get_subdirectories = lambda directory_path: [item for item in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, item))]
get_files_in_dir = lambda directory_path: [item for item in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, item))]

def run_command_with_timeout(command: str, timeout: int) -> bool:
    """Runs an OS process and terminates the process after a time limit is reached

    Arguments:
    command -- the command string to execute
    timeout -- the number of seconds to run command before termination
    """
    
    # Start the process
    process = subprocess.Popen(command)
    
    try:
        # Wait for the process to finish or timeout
        stdout, stderr = process.communicate(timeout=timeout)
        # Check if the process returned an error
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
        
    except subprocess.TimeoutExpired:
        # If timeout expired, kill the process
        process.kill()
        print("Process timed out")
        return True
    
    except Exception:
        print("Process failed")
        return True
    
    finally:
        # Close the process pipes
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()

    return False

def check_dir_exists(directory: str, error_message: str) -> None:
    """Checks if a directory exists and raises an error with a given message if not."""
    if not os.path.isdir(directory):
        raise AssertionError(error_message)

def check_file_exists(file_path: str, error_message: str) -> None:
    """Checks if a file exists and raises an error with a given message if not."""
    if not os.path.isfile(file_path):
        raise AttributeError(error_message)

def check_scene_exists(dataset_root: str, scene_dir: str, date: str) -> None:
    """Validates the existence of necessary directories for a given scene and date."""
    scene_path = os.path.join(dataset_root, scene_dir)
    date_path = os.path.join(scene_path, date)
    images_path = os.path.join(date_path, "images")
    check_dir_exists(scene_path, f"Scene {scene_dir} not found in dataset directory")
    check_dir_exists(date_path, f"Date {date} not found in scene directory {scene_dir}")

def check_transform_exists(dataset_root: str, scene_dir: str, date: str, transform_type: str) -> None:
    """Validates that the transform directory in the project exists."""
    transform_path = os.path.join(dataset_root, scene_dir, date, "transforms", transform_type)
    check_dir_exists(transform_path, f"Transform {transform_type} not found in scene {scene_dir}")

def check_images_exists(dataset_root: str, scene_dir: str, date: str, img_type: str) -> None:
    """Validates that the image directory in the project exists."""
    images_path = os.path.join(dataset_root, scene_dir, date, "images", img_type)
    check_dir_exists(images_path, f"Image type {img_type} not found in scene {scene_dir}")

def check_args(args) -> None:
    """Validates the command-line arguments."""

    # Checks that the project is valid 
    check_dir_exists(args.dataset_root, "Dataset directory is not a valid absolute path")
    check_scene_exists(args.dataset_root, args.scene_dir, args.date)
    check_transform_exists(args.dataset_root, args.scene_dir, args.date, args.transform_type)
    check_images_exists(args.dataset_root, args.scene_dir, args.date, args.img_type)

    if args.cam_path:
        check_file_exists(args.cam_path, "Camera Path does not exist")

    if args.render_video and not args.cam_path:
        raise AttributeError("Camera Path cannot be none for rendering a video")

    if len(args.bounding_box) != 6:
        raise AttributeError("Bounding Box must have exactly 6 values")

    try:
        args.bounding_box = [float(x) for x in args.bounding_box]
    except ValueError:
        raise AttributeError("Bounding Box must contain float values")

    if args.timestamp < 0:
        raise AttributeError("Timestamp must be a natural number")

    if args.num_points < 0:
        raise AttributeError("Number of points must be above 0 for point cloud/mesh generation")

    if args.num_faces < 0:
        raise AttributeError("Number of points must be above 0 for point cloud/mesh generation")

    if args.num_epochs < 0:
        raise AttributeError("Number of epochs must be above 0 for valid training")

def render_command(checkpoint_file: str, output_path: str, camera_path: str = None, video: bool = False) -> str:
    """Returns a command to render images or video."""
    if video:
        return f"ns-render camera-path --load-config {checkpoint_file} --camera-path-filename {camera_path} --output-path {output_path}"
    
    return f"ns-render dataset --load-config {checkpoint_file} --output-path {output_path} --image-format png"

def export_command(checkpoint_file: str, output_dir: str, num_points: int = 1000000, num_faces: int = 50000, bounding_box: list = None, mesh: bool = False) -> str:
    """Creates a command to export a point cloud or mesh."""
    if bounding_box is None:
        bounding_box = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]

    if mesh:
        return f"ns-export poisson --load-config {checkpoint_file} --output-dir {output_dir} --target-num-faces {num_faces} --num-pixels-per-side 2048 --normal-method open3d" \
               f" --num-points {num_points} --remove-outliers True --use-bounding-box True --bounding-box-min {bounding_box[0]} {bounding_box[2]} {bounding_box[4]}" \
               f" --bounding-box-max {bounding_box[1]} {bounding_box[3]} {bounding_box[5]}"
    
    return f"ns-export pointcloud --load-config {checkpoint_file} --output-dir {output_dir} --num-points {num_points} --remove-outliers True" \
               f" --normal-method open3d --use-bounding-box True --bounding-box-min {bounding_box[0]} {bounding_box[2]} {bounding_box[4]}" \
               f" --bounding-box-max {bounding_box[1]} {bounding_box[3]} {bounding_box[5]}"

def evaluate_model(checkpoint_file: str, output_path: str) -> str:
    """Creates a command to evaluate the model."""
    return f"ns-eval --load-config {checkpoint_file} --output-path {output_path}"

def export_gaussians(checkpoint_file: str, output_dir: str) -> str:
    """Creates a command to export Gaussian splatting."""
    return f"ns-export gaussian-splat --load-config {checkpoint_file} --output-dir {output_dir}"

def view_model(checkpoint_file: str) -> str:
    """Creates a command to view the model in real-time."""
    return f"ns-viewer --load-config {checkpoint_file}"

def train_model(transform_path: str, experiment_name: str, model: str, output_dir: str, img_path: str = None,
                num_epochs:int = 300000, timestamp: str = "1", eval_type="filename", is_3dgs: bool = False) -> str:
    """Creates a command to train a NeRF or Gaussian Splatting model."""
    if is_3dgs:
        return f"ns-train {model} --data {transform_path} --experiment-name {experiment_name} --output-dir {output_dir} --logging.steps-per-log 5000 --max_num_iterations {num_epochs}" \
               f" --viewer.quit-on-train-completion True --timestamp {timestamp} --pipeline.model.cull-alpha-thresh 0.005 --pipeline.model.continue-cull-post-densification False" \
               f" colmap --orientation-method none --auto-scale-poses False --center-method poses --eval-mode {eval_type}" \
               f" --images-path {img_path} --colmap-path sparse\\0 --downscale-factor 1 --assume-colmap-world-coordinate-convention False"
    
    return f"ns-train {model} --data {transform_path} --experiment-name {experiment_name} --output-dir {output_dir} --logging.steps-per-log 5000" \
               f" --max_num_iterations {num_epochs} --viewer.quit-on-train-completion True --timestamp {timestamp}" \
               f" nerfstudio-data --orientation-method none --auto-scale-poses False --center-method poses --eval-mode {eval_type}"

def get_reconstruction_dir(model: str) -> str:
    """Determines the reconstruction directory based on the model type."""
    if model.lower().strip() in GAUSSIAN_SPLATTING_MODELS:
        return "gaussian-splatting"
    return "nerf"

def run_experiment(args) -> None:
    """Runs the experiment based on provided arguments."""
    check_args(args)

    scene_date_path = os.path.join(args.dataset_root, args.scene_dir, args.date)
    reconstruction_path = os.path.join(scene_date_path, get_reconstruction_dir(args.model))
    exports_path =  os.path.join(scene_date_path, "exports")

    os.makedirs(reconstruction_path, exist_ok=True)
    os.makedirs(exports_path, exist_ok=True)

    transform_path = os.path.join(scene_date_path, "transforms", args.transform_type)
    images_path = os.path.join(scene_date_path, "images", args.img_type)

    # Use the filename evaluation type if the images are in this format, other use fraction (which will work with any dataset)
    eval_type = "filename" if any("_eval" in img_name for img_name in get_files_in_dir(images_path)) else "fraction"

    checkpoint_dir = os.path.join(reconstruction_path, args.experiment_name, args.model, str(args.timestamp))
    checkpoint_path = os.path.join(checkpoint_dir, "config.yml")

    # Ensures that the weights are in the correct format to match the filename
    num_steps = f"{(args.num_epochs-1):09}"
    saved_weights_path = os.path.join(checkpoint_dir, "nerfstudio_models", f"step-{num_steps}.ckpt")

    if args.train_model:
        print("Training model")
        
        if (not os.path.exists(checkpoint_path) or not os.path.exists(saved_weights_path)) or args.override:
            if run_command_with_timeout(train_model(transform_path, args.experiment_name, args.model, reconstruction_path, images_path,
                                                    num_epochs=args.num_epochs, timestamp=args.timestamp, eval_type=eval_type, 
                                                    is_3dgs=(args.model in GAUSSIAN_SPLATTING_MODELS)), args.timeout):
                raise Exception("Model training failed or timed out")
        else:
            print(f"SKIPPING: This model has already been fully trained")

    # If the checkpoint config file is not found, then it is not possible to perform any other operations in the model
    if not os.path.isfile(checkpoint_path):
        print(f"Cannot continue as no config file found for this {args.scene_dir} {args.date}")
        return

    # While models that are not fully trained can still be used, it is recommended that the model is fully trained (with max epochs)
    if not os.path.exists(saved_weights_path):
        print("WARNING: Cannot locate weights with the given number of epochs, model might need more training") 

    if args.eval:
        print("Evaluating model")
        
        eval_output_path = os.path.join(checkpoint_dir, "eval_nerfacto.json")
        
        if not os.path.exists(eval_output_path) or args.override:
            run_command_with_timeout(evaluate_model(checkpoint_path, eval_output_path), args.timeout)
        else:
            print(f"SKIPPING: An evaluation file exists for {args.experiment_name}")

    if args.render_image or args.eval_masked:
        print("Rendering images")
        
        image_output_path = os.path.join(exports_path, "render_imgs", args.experiment_name)
        os.makedirs(image_output_path, exist_ok=True)
        
        if (not os.listdir(image_output_path) or not os.path.exists(os.path.join(image_output_path, "test"))) or args.override:
            run_command_with_timeout(render_command(checkpoint_path, image_output_path), args.timeout)
        else:
            print(f"SKIPPING: Rendered images already exists for {args.experiment_name}")

    if args.eval_masked:
        print("Evaluating with mask")
        
        evaluation_masked_path = os.path.join(checkpoint_dir, "eval_masked.json")
        rendered_imgs_path = os.path.join(image_output_path, "test", "rgb")
        
        if (not os.path.isfile(evaluation_masked_path) and os.path.exists(rendered_imgs_path)) or args.override:
            mask_imgs_path = "mask" if get_reconstruction_dir(args.model) != "gaussian-splatting" else "undistorted_mask"
            mask_imgs_path = os.path.join(scene_date_path, "images", mask_imgs_path)

            if os.path.exists(mask_imgs_path) or args.override:
                try:
                    calculate_masked_psnr_set(rendered_imgs_path, images_path, mask_imgs_path, evaluation_masked_path)
                except Exception as e:
                    print(e)
                    print("Failed to calulate masked PSNR")
            else:
                print("Failed to calculate masked evaluation as mask images do not exist")
        else:
            print("SKIPPING: Mask evaluation results already exist or evaluation images have not been rendered correctly")

    if args.render_video:
        print("Rendering Video")
        
        video_output_path = os.path.join(exports_path, "render_video")
        os.makedirs(video_output_path, exist_ok=True)

        render_video_path = os.path.join(video_output_path, args.experiment_name + ".mp4")
        if not os.path.exists(render_video_path):
            run_command_with_timeout(render_command(checkpoint_path, render_video_path, camera_path=args.cam_path, video=True), args.timeout)
        else:
            print("SKIPPING: Render Video already exists")

    if args.generate_mesh:
        print("Generating Mesh")
        
        export_output_path = os.path.join(exports_path, "mesh", args.experiment_name)
        os.makedirs(export_output_path, exist_ok=True)
        if not os.path.exists(os.path.join(export_output_path, f"{args.experiment_name}.ply")) or args.override:
            err = run_command_with_timeout(export_command(checkpoint_path, export_output_path, num_points=args.num_points,
                                                                 num_faces=args.num_faces, bounding_box=args.bounding_box, mesh=True), args.timeout)

            if not err:
                os.rename(os.path.join(export_output_path, "poisson_mesh.ply"), os.path.join(export_output_path, f"{args.experiment_name}.ply"))
        else:
            print(f"SKIPPING: Mesh already exists for {args.experiment_name}")

    if args.generate_pointcloud:
        print("Generating Pointcloud")
        
        export_output_path = os.path.join(exports_path, "pointcloud")
        os.makedirs(export_output_path, exist_ok=True)

        if not os.path.exists(os.path.join(export_output_path, f"{args.experiment_name}.ply")) or args.override:
            err = run_command_with_timeout(export_command(checkpoint_path, export_output_path, num_points=args.num_points,
                                                                 bounding_box=args.bounding_box, mesh=False), args.timeout)

            if not err:
                os.rename(os.path.join(export_output_path, "point_cloud.ply"), os.path.join(export_output_path, f"{args.experiment_name}.ply"))
        else:
            print(f"SKIPPING: Pointcloud already exists for {args.experiment_name}")
            
    if args.export_gaussians:
        print("Exporting Gaussians")

        if not os.path.exists(os.path.join(checkpoint_dir, f"splat.ply")) or args.override:
            run_command_with_timeout(export_gaussians(checkpoint_path, checkpoint_dir), args.timeout)
        else:
            print(f"SKIPPING: Gaussians have already been exported")

    if not args.skip_view:
        print("Viewing Trained Model")
        
        if run_command_with_timeout(view_model(checkpoint_path), args.timeout):
            print("Viewing model failed or timed out")

if __name__ == "__main__":
    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True, help="Path to configuration file containing a list of argument values")

    parser.add_argument("--dataset_root", type=str, default="", help="The path to the dataset directory")
    parser.add_argument("--experiment_name", type=str, default="", help="Name of the experiment")
    
    parser.add_argument("--scene_dir", type=str, default="", help="Scene directory to process")
    parser.add_argument("--date", type=str, default="",help="The date for the specific scene directory")
    
    parser.add_argument("--transform_type", type=str, default="adjusted", help="Type of transformation to use")
    parser.add_argument("--img_type", type=str, default="rgb", help="Type of images to use")
    
    parser.add_argument("--timestamp", type=int, default=1, help="Timestamp for the process")
    parser.add_argument("--num_epochs", type=int, default=30000, help="Number of epochs for training")
    
    parser.add_argument("--num_points", type=int, default=1000000, help="Number of points for the point cloud or mesh")
    parser.add_argument("--num_faces", type=int, default=50000, help="The number of faces to generate for the mesh")
    parser.add_argument("--bounding_box", nargs=6, default=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], help="Bounding box for point cloud or mesh export")
    
    parser.add_argument("--model", type=str, default="nerfacto", help="The NeRFStudio model to use for training")
    
    parser.add_argument("--cam_path", type=str, help="Path to the camera file for rendering videos")
    
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout for each command execution in seconds")
    
    parser.add_argument("--train_model", action="store_true", help="Flag to induce training")
    
    parser.add_argument("--render_image", action="store_true", help="Flag to indicate if images should be rendered")
    parser.add_argument("--render_video", action="store_true", help="Flag to indicate if video should be rendered")
            
    parser.add_argument("--generate_pointcloud", action="store_true", help="Flag to indicate if the model should be exported to a pointcloud")
    parser.add_argument("--generate_mesh", action="store_true", help="Flag to indicate if the model should be exported to a pointcloud")
            
    parser.add_argument("--eval", action="store_true", help="Flag to indicate if the model should be evaluated")
    parser.add_argument("--eval_masked", action="store_true", help="Flag to indicate if the model should be evaluated using the masked PSNR method")
            
    parser.add_argument("--export_gaussians", action="store_true", help="Flag to indicate if Gaussians should be exported")
    parser.add_argument("--skip_view", action="store_true", help="Flag to indicate if the model should be viewed")

    parser.add_argument("--override", action="store_true", help="Flag to indicate if previous data should be overridden")

    args = parser.parse_args()

    if args.dataset_root == "":
        args.dataset_root = os.path.dirname(os.path.abspath(__file__))

    if args.experiment_name == "":
        args.experiment_name = args.transform_type

    # If a scene directory is given, then just execute this
    if args.scene_dir != "":
        run_experiment(args)
        exit(0)
        
    print("No scene directory given- iterating through all scene directories")
    print()

    dataset_scenes = get_subdirectories(args.dataset_root)

    # Iterate through every scene
    for scene_dir in dataset_scenes:

        args.scene_dir = scene_dir

        dates = get_subdirectories(os.path.join(args.dataset_root, scene_dir))

        # Iterate through every date in the current scene
        for date in dates:
            args.date = date

            transform_dir = os.path.join(args.dataset_root, scene_dir, date, "transforms")
            transforms = get_subdirectories(transform_dir)

            # Iterate through all transforms in the current date for the current scene
            for transform in transforms:
                args.transform_type = transform
                args.experiment_name = transform
                    
                if not os.path.isfile(os.path.join(transform_dir, args.transform_type, "transforms.json")):
                    args.model = "splatfacto"
                    args.img_type = "undistorted"

                run_experiment(args)
                    
