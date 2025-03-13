# Ultrasound-VID

Video instance detection for ultrasound videos.

The whole project is based on `detectron2`, which is used as a package.

This project is also an example of how to use custom dataset, dataloader and model with `detectron2`.

# Requirments

## Installation

```
# install anaconda3 with python3.9
conda create -n <env_name> python=3.9
conda activate <env_name>

# use tuna source for pypi (optional)
python -m pip install --upgrade pip

# install packages
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install pandas
pip install path
pip install sqlmodel
pip install coloredlogs
pip install opencv-python
pip install mmcv
pip install prettytable
pip install fire
pip install scipy
pip install timm
pip install pytorch_metric_learning

# install ultrasound_vid
pip install -e .
cd ultrasound_vid/modeling/ops
export CUDA_HOME=<cuda_home_path>
sh make.sh

# issue shooting
# AttributeError: module 'distutils' has no attribute 'version'
pip uninstall setuptools
pip install setuptools==58.0.4
```

## Dataset Preparation

We provide high-quality labels in `bus_data_cva_new/trainval` and `bus_data_cva_new/test`.

Please download JPG images from https://github.com/jhl-Det/CVA-Net and organize images and labels as follows.

```
ultrasound_vid/
  datasets/
    bus_data_cva_new/
      videos/
        1dc9ca2f1748c2ec/
          0.jpg
          1.jpg
          ...
        2a2716116cf2a1da/
          0.jpg
          1.jpg
          ...
      trainval/
        20230309.csv
        1dc9ca2f1748c2ec.json
        ...
      test/
        20230309.csv
        2c977dd2ec9d714a.json
        ...
```