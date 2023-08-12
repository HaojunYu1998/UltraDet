# UltraDet

This is the official implementation of [Mining Negative Temporal Contexts For False Positive Suppression In Real-Time Ultrasound Lesion Detection](https://arxiv.org/abs/2305.18060). We provide a real-time breast ultrasound detection solution to **reduce about 50% FPs at a recall rate of 0.90**. Our paper has been early accepted by MICCAI 2023!

[[Arxiv]](https://arxiv.org/abs/2305.18060) [[PDF]](https://arxiv.org/pdf/2305.18060.pdf) [[Code]](https://github.com/HaojunYu1998/UltraDet)

<div align="center">
    <img src="assets/teaser.png" width="100%"/>
</div>

## Paper Abstract

During ultrasonic scanning processes, real-time lesion detection can assist radiologists in accurate cancer diagnosis. However, this essential task remains challenging and underexplored. General-purpose real-time object detection models can mistakenly report obvious false positives (FPs) when applied to ultrasound videos, potentially misleading junior radiologists. One key issue is their failure to utilize negative symptoms in previous frames, denoted as negative temporal contexts (NTC). To address this issue, we propose to extract contexts from previous frames, including NTC, with the guidance of inverse optical flow. By aggregating extracted contexts, we endow the model with the ability to suppress FPs by leveraging NTC. We call the resulting model UltraDet. The proposed UltraDet demonstrates significant improvement over previous state-of-the-arts and achieves real-time inference speed.

## Experiment Results

The NTCA module leverages RoI-level NTC which are crucial for radiologists but ignored in previous works, thereby effectively improving the detection performance in a reliable and interpretable way. We plug the NTCA module into a basic real-time detector to form UltraDet. Experiments on CVA-BUS dataset demonstrate that UltraDet, with real-time inference speed, significantly outperforms previous works, reducing about 50% FPs at a recall rate of 0.90.


<div align="center">
    <img src="assets/visualization.png" width="100%"/>
</div>

<center>
<table>
  <caption>Quantitative results of real-time lesion detection on CVA-BUS.</caption>
  <thead>
    <tr>
      <th>Model</th>
      <th>Type</th>
      <th>Pr<sub>80</sub></th>
      <th>Pr<sub>90</sub></th>
      <th>FP<sub>80</sub></th>
      <th>FP<sub>90</sub></th>
      <th>AP<sub>50</sub></th>
      <th>R@16</th>
      <th>FPS</th>
      <th>Model Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="10" style="text-align: center;">One-Stage Detectors</td>
    </tr>
    <tr>
      <td>YOLOX</td>
      <td>Image</td>
      <td>69.7<sub>3.7</sub></td>
      <td>43.4<sub>7.7</sub></td>
      <td>23.8<sub>4.8</sub></td>
      <td>87.6<sub>24.5</sub></td>
      <td>80.4<sub>1.6</sub></td>
      <td>97.5<sub>0.5</sub></td>
      <td><b>59.8</b></td>
      <td style="text-align: center;">-</td>
    </tr>
    <tr>
      <td>RetinaNet</td>
      <td>Image</td>
      <td>75.7<sub>2.5</sub></td>
      <td>57.2<sub>2.9</sub></td>
      <td>9.3<sub>2.0</sub></td>
      <td>32.8<sub>6.5</sub></td>
      <td>84.5<sub>1.0</sub></td>
      <td>95.1<sub>0.6</sub></td>
      <td>53.6</td>
      <td style="text-align: center;">-</td>
    </tr>
    <tr>
      <td>FCOS</td>
      <td>Image</td>
      <td>87.2<sub>2.2</sub></td>
      <td>72.2<sub>5.1</sub></td>
      <td>11.0<sub>2.4</sub></td>
      <td>23.0<sub>3.7</sub></td>
      <td>89.5<sub>1.4</sub></td>
      <td>98.8<sub>0.3</sub></td>
      <td>56.1</td>
      <td style="text-align: center;">-</td>
    </tr>
    <tr>
      <td>DeFCN</td>
      <td>Image</td>
      <td>81.5<sub>1.8</sub></td>
      <td>67.5<sub>2.3</sub></td>
      <td>21.1<sub>3.2</sub></td>
      <td>33.4<sub>4.3</sub></td>
      <td>86.4<sub>1.3</sub></td>
      <td><b>99.3</b><sub>0.3</sub></td>
      <td>51.2</td>
      <td style="text-align: center;">-</td>
    </tr>
    <tr>
      <td>Track-YOLO</td>
      <td>Video</td>
      <td>75.1<sub>2.7</sub></td>
      <td>47.0<sub>3.1</sub></td>
      <td>18.1<sub>1.9</sub></td>
      <td>74.2<sub>14.7</sub></td>
      <td>80.1<sub>1.0</sub></td>
      <td>94.7<sub>0.9</sub></td>
      <td>46.0</td>
      <td style="text-align: center;">-</td>
    </tr>
    <tr>
      <td colspan="10" style="text-align: center;">DETR-Based Detectors</td>
    </tr>
  <tr>
    <td>DeformDETR</td>
    <td>Image</td>
    <td>90.1<sub>3.2</sub></td>
    <td>72.7<sub>10.6</sub></td>
    <td>5.6<sub>2.2</sub></td>
    <td>37.8<sub>20.9</sub></td>
    <td>90.5<sub>2.0</sub></td>
    <td>98.7<sub>0.3</sub></td>
    <td>33.8</td>
    <td style="text-align: center;">-</td>
  </tr>
  <tr>
    <td>TransVOD</td>
    <td>Video</td>
    <td>92.5<sub>2.2</sub></td>
    <td>77.5<sub>7.2</sub></td>
    <td>3.1<sub>1.3</sub></td>
    <td>23.7<sub>11.5</sub></td>
    <td>90.1<sub>1.8</sub></td>
    <td>98.4<sub>0.4</sub></td>
    <td>24.2</td>
    <td style="text-align: center;">-</td>
  </tr>
  <tr>
    <td>CVA-Net</td>
    <td>Video</td>
    <td>92.3<sub>2.6</sub></td>
    <td>80.2<sub>6.1</sub></td>
    <td>4.7<sub>2.6</sub></td>
    <td>19.6<sub>5.6</sub></td>
    <td><strong>91.6</strong><sub>1.9</sub></td>
    <td>98.6<sub>0.8</sub></td>
    <td>23.1</td>
    <td style="text-align: center;">-</td>
  </tr>
  <tr>
    <td>PTSEFormer</td>
    <td>Video</td>
    <td>93.3<sub>1.9</sub></td>
    <td>85.4<sub>6.0</sub></td>
    <td>2.8<sub>1.1</sub></td>
    <td>12.5<sub>9.8</sub></td>
    <td>91.5<sub>1.6</sub></td>
    <td>97.9<sub>1.2</sub></td>
    <td>9.1</td>
    <td style="text-align: center;">-</td>
  </tr>
    <tr>
      <td colspan="10" style="text-align: center;">FasterRCNN-Based Detectors</td>
    </tr>
    <td>FasterRCNN</td>
    <td>Image</td>
    <td>91.3<sub>0.9</sub></td>
    <td>75.2<sub>3.6</sub></td>
    <td>6.9<sub>1.4</sub></td>
    <td>34.4<sub>6.7</sub></td>
    <td>88.0<sub>1.4</sub></td>
    <td>92.4<sub>1.0</sub></td>
    <td>49.2</td>
    <td style="text-align: center;">-</td>
  </tr>
  <tr>
    <td>RelationNet</td>
    <td>Image</td>
    <td>91.4<sub>1.3</sub></td>
    <td>79.2<sub>2.9</sub></td>
    <td>6.2<sub>2.0</sub></td>
    <td>24.4<sub>5.6</sub></td>
    <td>87.6<sub>1.7</sub></td>
    <td>92.4<sub>0.9</sub></td>
    <td>42.7</td>
    <td style="text-align: center;">-</td>
  </tr>
  <tr>
    <td>FGFA</td>
    <td>Video</td>
    <td>92.9<sub>1.5</sub></td>
    <td>82.2<sub>4.1</sub></td>
    <td>4.4<sub>1.6</sub></td>
    <td>13.3<sub>3.7</sub></td>
    <td>90.5<sub>1.1</sub></td>
    <td>93.6<sub>0.9</sub></td>
    <td>33.8</td>
    <td style="text-align: center;">-</td>
  </tr>
  <tr>
    <td>SELSA</td>
    <td>Video</td>
    <td>91.6<sub>1.7</sub></td>
    <td>80.2<sub>2.5</sub></td>
    <td>7.5<sub>1.5</sub></td>
    <td>23.3<sub>5.5</sub></td>
    <td>89.2<sub>1.1</sub></td>
    <td>92.6<sub>0.8</sub></td>
    <td>43.8</td>
    <td style="text-align: center;">-</td>
  </tr>
  <tr>
    <td>MEGA</td>
    <td>Video</td>
    <td>93.9<sub>1.5</sub></td>
    <td>86.9<sub>2.3</sub></td>
    <td>3.1<sub>1.7</sub></td>
    <td>11.7<sub>3.0</sub></td>
    <td>90.9<sub>1.0</sub></td>
    <td>93.6<sub>0.7</sub></td>
    <td>40.2</td>
    <td style="text-align: center;">-</td>
  </tr>
  <tr>
    <td>BasicDet (RDN)</td>
    <td>Video</td>
    <td>92.4<sub>1.0</sub></td>
    <td>83.6<sub>2.2</sub></td>
    <td>3.8<sub>1.2</sub></td>
    <td>13.4<sub>3.2</sub></td>
    <td>88.7<sub>1.4</sub></td>
    <td>92.7<sub>0.6</sub></td>
    <td>42.2</td>
    <td style="text-align: center;">-</td>
  </tr>
  <tr>
    <td>UltraDet (Ours)</td>
   <td>Video</td>
    <td><strong>95.7<sub>1.2</sub></strong></td>
    <td><strong>90.8<sub>1.4</sub></strong></td>
    <td><strong>1.9<sub>0.4</sub></strong></td>
    <td><strong>5.7<sub>1.6</sub></strong></td>
    <td><strong>91.6<sub>1.6</sub></strong></td>
    <td>93.8<sub>1.3</sub></td>
    <td>30.4</td>
    <td style="text-align: center;"><a href="https://github.com/HaojunYu1998/UltraDet/releases/download/Publish/ultradet.pth">Link</a></td>
  </tr>
</table>
</center>

## Citing UltraDet

If you find UltraDet useful in your research, please cite our paper:

```bibtex
@article{yu2023mining,
  title={Mining Negative Temporal Contexts For False Positive Suppression In Real-Time Ultrasound Lesion Detection},
  author={Yu, Haojun and Li, Youcheng and Wu, QuanLin and Zhao, Ziwei and Chen, Dengbo and Wang, Dong and Wang, Liwei},
  journal={arXiv preprint arXiv:2305.18060},
  year={2023}
}
```

## Requirments

The whole project is based on [detectron2](https://github.com/facebookresearch/detectron2).

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
pip install opencv-p
pip install mmcv
pip install prettytable
pip install fire
pip install scipy
pip install timm

# install ultrasound_vid
git clone https://github.com/HaojunYu1998/UltraDet.git
cd UltraDet
pip install -e .

# issue shooting
# AttributeError: module 'distutils' has no attribute 'version'
pip uninstall setuptools
pip install setuptools==58.0.4
```

## Checkpoints

We are pleased to share the weights of the models required for UltraDet inference. You can access them through the following links:

<center>
  <table>
    <thead>
      <tr>
        <th>Weigths</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><a href="https://github.com/HaojunYu1998/UltraDet/releases/download/Publish/flownet.ckpt">flownet.ckpt</a></td>
      </tr>
      <tr>
        <td><a href="https://github.com/HaojunYu1998/UltraDet/releases/download/Publish/r34.pkl">r34.pkl</a></td>
      </tr>
      <tr>
        <td><a href="https://github.com/HaojunYu1998/UltraDet/releases/download/Publish/ultradet.pth">ultradet.pth</a></td>
      </tr>
    </tbody>
  </table>
</center>

After downloading the weights, please store them in the following path:

```
UltraDet/
  pretrained_models/
    flownet.ckpt
    r34.pkl
    ultradet.pth
```

## Dataset Preparation

We provide high quality annotation files. You can access them through the following links:

<center>
<table>
  <thead>
    <tr>
      <th>Json File</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/HaojunYu1998/UltraDet/releases/download/Publish/test.json">test.json</a></td>
    </tr>
    <tr>
      <td><a href="https://github.com/HaojunYu1998/UltraDet/releases/download/Publish/trainval.json">trainval.json</a></td>
    </tr>
  </tbody>
</table>
</center>

Please download PNG images from [CVA-Net](https://github.com/jhl-Det/CVA-Net) and arrange the images and labels in the COCO format. If you encounter any data-related issues, please contact us via ``haojunyu@pku.edu.cn``.

**Note: The authors of CVA-Net dataset have made corrections to the benign/malignant classification annotations** (please see [CVA-Net](https://github.com/jhl-Det/CVA-Net)). To use our high-quality annotations, please first correct the video and image location in ``rawframes/``.

```
UltraDet/
  datasets/
    bus_data_cva/
      rawframes/
        benign/
          2cda21c3aab26332/
            000000.png
            000001.png
            ...
          ...
        malignant/
          1dc9ca2f1748c2ec/
            000000.png
            000001.png
            ...
          ...
      test.json
      trainval.json
```

## Training and Inference

You can train or infer with UltraDet by running the following code:

```shell
# Training
python train_net.py --num-gpus {GPU_NUM} \
--config-file {CONFIG_FILES}

# Validation
python train_net.py --num-gpus {GPU_NUM} --eval-only --resume \
--config-file {CONFIG_FILES}

# Test
python train_net.py --num-gpus {GPU_NUM} --eval-only  --resume \
--config-file {CONFIG_FILES} \
DATASETS.SPLIT "test" \
DATASETS.NUM_FOLDS 1 \
# MODEL.WEIGHTS 'pretrained_models/ultradet.pth' # If you wish to load the pre-trained weights of UltraDet that we have released, please include this line of code
```

