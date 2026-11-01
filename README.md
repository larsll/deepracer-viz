# AWS DeepRacer Video Visualizations
Repository containing tools to visualize AWS DeepRacer (training) runs. Currently only a visualization based on GradCam is supported.

# Setup
## Requirements
- Python 3
- [Poetry](https://github.com/sdispater/poetry)

## Installation
First prepare the Python enviroment using `poetry install`.
Available tools can be found in the `tools` folder. Before running a tool make sure the virtual environment is activated using `poetry shell`.

## Downloading your model
You can obtain your model graph in the form of a `.pb` file either from S3 or from the DeepRacer console.
- S3 -> Go to the S3 folder in which the training run is stored. Here you can download the model files per checkpoint.
- Console -> Select your training run and choose `Download model`. Inside the obtained `.tar.gz` you can find your `model.pb` file. 

# Visualization tools

## Kinesis Downloader
Can be used to view and store the live feed of a training job. This is the same feed that is shown in the AWS DeepRacer console.

`python download_kinesis.py <Kinesis Video Stream Name> -o <output_file.mp4>`


## GradCam
Overlays a class activation map over an input video, showing which parts of the camera image drove the model's decision.
By default the action selected by the model in each frame is explained. For discrete action spaces a specific action
(index as in `model_metadata.json` or the DeepRacer console) can be explained instead with `-a`. For continuous action
spaces choose between explaining `steering` (in the direction the car is turning) or `speed` with `--continuous-target`.

![GradCam example](img/example-gradcam.gif)

`drviz gradcam -i <input_file.mp4> -o <output_file.mp4> -m <model.pb> -mm <model_metadata.json> [--method <method>] [-a <action index>]`

Available methods (`--method`):
- `hirescam` (default) - [HiResCAM, Draelos & Carin](https://arxiv.org/abs/2011.08891). Element-wise gradient × activation; only highlights regions that actually contributed to the score.
- `gradcam` - [Grad-CAM, Selvaraju et al.](https://arxiv.org/abs/1610.02391).
- `gradcampp` - [Grad-CAM++, Chattopadhay et al.](https://arxiv.org/abs/1710.11063).
- `xgradcam` - [XGrad-CAM, Fu et al.](https://arxiv.org/abs/2008.02312).
- `layercam` - [LayerCAM, Jiang et al.](https://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf). Fuses the maps of all convolutional layers.

Use `--layers last|all` to select whether the map is computed on the last convolutional layer (default) or averaged over all convolutional layers (default for `layercam`).
