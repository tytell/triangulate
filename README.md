# Scripts for calibration and 3D triangulation

Wrappers and diagnostics for calibrating using a ChaRuCo board and triangulating points in 3D using multiple cameras. Assumes that points have been tracked and exported from (sleap.ai)[https://sleap.ai/].
All the actual analysis is through [aniposelib](https://github.com/lambdaloop/aniposelib).

You should have a set of axes calibration videos or images, in which you take an image of an object that defines the origin and X, Y, and Z axes from each camera. Use the [axes.json](./axes.json) skeleton to identify the axes points in Sleap. Then just export the points from Sleap. (There's no need to train a network or perform any inference for axes points)

## Installation

You should be able to install the necessary packages with the [env-triangulate.yml](./env-triangulate.yml) file.

``
$ conda env create -f env-triangulate.yml
``

## 1. Modify the YAML configuration file

Copy [triangulate.yml](./triangulate.yml) to your working directory. These are the parameters in the configuration file that should be modified. Modify the others only if you know what you're doing!

### Main parameters
* `base_path`: This should be the full path to your main working directory. Make sure to include the drive (e.g., "D:") on Windows machines.
* `project_name`: A name for your project (**without spaces**). This will be added into the names of the output files
* `user`: Your name (**without spaces**). Can be left blank

### Calibration parameters
* `calibration_videos`: List of the videos from each camera that contain the images of the ChaRuCo board
* `camera_names`: Names of the cameras, in the same order as the files in `calibration_videos`

## Axes parameters
* `axes_files_path`: Name of the folder relative to your base path that contains the axes points files, as exported from Sleap.
* `axes_files`: List of the points from Sleap that define your axes

## Video triangulation parameters
* `points_files_path`: Name of the folder relative to your base path that contains the tracked points files, as exported from Sleap.
* `points_files_list`: Name of a CSV file that contains each trial and the names of the matching exported points files from each camera. See [points_files_list.csv](./points_files_list.csv) for an example.
* `output_file`: Name of the output file for the triangulated points. Note that the final output file will have the name of the project, the user name (if given), and the current date before this name. (For example, the template YAML file would produce a final output file 'triangulate-Eric-2025-03-18T1136-data3d.csv')

## 2. Run the main script

Make sure that you've activated the conda environment.

``
$ conda activae triangulate
``

Then run the main script with

``
$ python do_triangulate.py
``
